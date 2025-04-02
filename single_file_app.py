# single_file_app.py

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber # For PDF viewer rendering
from openai import OpenAI
import os
import asyncio
from typing import List, Dict, Optional, Tuple, Any
import logging
from langchain.schema.messages import SystemMessage, HumanMessage # Assuming still used by DocumentAnalyzer internally
import json
import base64
import re
from io import BytesIO
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime
import concurrent.futures
import threading
import queue
import atexit
from dotenv import load_dotenv
import google.generativeai as genai

# --- Dependencies for Fuzzy Matching & Word Processing ---
from thefuzz import fuzz
from docx import Document as DocxDocument # Renamed to avoid conflict

# --- NLTK for potential fallback chunking (less preferred now) ---
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
# from nltk.tokenize import sent_tokenize

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# ****** SET PAGE CONFIG HERE (First Streamlit command) ******
st.set_page_config(layout="wide", page_title="SmartDocs Analysis")
# ****** END SET PAGE CONFIG ******


MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 4))
ENABLE_PARALLEL = os.environ.get('ENABLE_PARALLEL', 'true').lower() == 'true'
FUZZY_MATCH_THRESHOLD = 88 # Adjust this threshold (0-100)

# --- Helper Functions ---

def normalize_text(text: Optional[str]) -> str:
    """Normalize text for comparison: lowercase, strip, whitespace."""
    if not text:
        return ""
    text = str(text)
    text = text.lower() # Case-insensitive matching
    text = re.sub(r'\s+', ' ', text) # Normalize whitespace
    # Optional: Remove simple punctuation if causing issues - use cautiously
    # text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def remove_markdown_formatting(text: Optional[str]) -> str:
    """Removes common markdown formatting."""
    if not text: return ""
    text = str(text)
    # Basic bold, italics, code
    text = re.sub(r'\*(\*|_)(.*?)\1\*?', r'\2', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Basic headings, blockquotes, lists
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def get_base64_encoded_image(image_path: str) -> Optional[str]:
    """Get base64 encoded image."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None

def run_async(coro):
    """Helper function to run async code in a thread-safe manner."""
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Thread-safe counter class
class Counter:
    def __init__(self, initial_value=0):
        self._value = initial_value
        self._lock = threading.Lock()
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    def value(self):
        with self._lock:
            return self._value

# --- AI Analyzer Class (Adapted from user's ai_analyzer.py) ---
_thread_local = threading.local()

class DocumentAnalyzer:
    def __init__(self):
        pass # Lazy initialization

    def _ensure_client(self):
        """Ensure that the Google client is initialized for the current thread."""
        if not hasattr(_thread_local, 'google_client'):
            try:
                # Use st.secrets for the Google API Key
                if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
                     api_key = st.secrets["GOOGLE_API_KEY"]
                     logger.info("Using Google API key from Streamlit secrets.")
                else:
                    # Fallback logic (e.g., environment variables)
                    api_key = os.getenv('GOOGLE_API_KEY')
                    if not api_key:
                         raise ValueError("Google API Key is missing. Check Streamlit secrets or GOOGLE_API_KEY environment variable.")
                    logger.info("Using Google API key from environment variable.")

                genai.configure(api_key=api_key)
                # Create the model instance - adjust model name as needed
                # See https://ai.google.dev/models/gemini
                _thread_local.google_client = genai.GenerativeModel('gemini-2.0-flash') # Or 'gemini-pro', etc.
                logger.info(f"Initialized Google GenAI client for thread {threading.current_thread().name} with model: {_thread_local.google_client.model_name}")


            except Exception as e:
                logger.error(f"Error initializing Google AI client: {str(e)}")
                raise
        return _thread_local.google_client

    async def _get_completion(self, messages: List[Dict[str, str]], model: str = "gemini-2.0-flash") -> str:
        """Helper method to get completion from the Google model."""
        # NOTE: Model name might need adjustment based on availability, e.g., "databricks-meta-llama-3-70b-instruct"
        try:
            client = self._ensure_client()
            # logger.debug(f"Formatted messages for API: {json.dumps(messages, indent=2)}") # Use DEBUG level

            history = []
            system_instruction = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    # Google prefers system instructions via GenerationConfig or start of history
                    system_instruction = content # Store it
                elif role == "user":
                    history.append({'role': 'user', 'parts': [{'text': content}]})
                elif role == "assistant":
                    history.append({'role': 'model', 'parts': [{'text': content}]})

            # Prepend system instruction if present
            if system_instruction:
                 # Method 1: Include in GenerationConfig (Recommended if API supports it well)
                 # config = genai.types.GenerationConfig(temperature=0.1, ...) # Add config
                 # response = await client.generate_content_async(history, generation_config=config, system_instruction=system_instruction)

                 # Method 2: Prepend to first user message (Less ideal, but common workaround)
                 if history and history[0]['role'] == 'user':
                      history[0]['parts'][0]['text'] = f"{system_instruction}\n\n---\n\n{history[0]['parts'][0]['text']}"
                 else: # Or add as a separate user message if history starts with model
                      history.insert(0, {'role': 'user', 'parts': [{'text': system_instruction}]})
                 logger.warning("Prepending system prompt to user message for Google API.")


            logger.info(f"Sending request to Google model: {client.model_name}")
            # logger.debug(f"Formatted history for Google API: {json.dumps(history, indent=2)}")

            # Use generate_content_async
            response = await client.generate_content_async(
                history,
                generation_config=genai.types.GenerationConfig(
                    # candidate_count=1, # Default is 1
                    # stop_sequences=['...'],
                    max_output_tokens=4096,
                    temperature=0.1,
                )
            )

            # Handle potential safety blocks or empty responses
            if not response.candidates:
                 safety_info = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'No specific feedback.'
                 logger.error(f"Google API returned no candidates. Possibly blocked. Feedback: {safety_info}")
                 raise ValueError(f"Google API returned no candidates. Content may have been blocked. Feedback: {safety_info}")

            content = response.text # Access text directly
            logger.info(f"Received response from Google model {client.model_name}")
            # logger.debug(f"Raw Google API response content: {content}")
            return content

        except Exception as e:
            logger.error(f"Error getting completion from Google model: {str(e)}", exc_info=True)
            # Attempt to get more detailed error if available
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                 logger.error(f"Google API Error Response Text: {e.response.text}")
            raise

            raise

    @property
    def output_schema_analysis(self) -> dict:
        """Defines the expected JSON structure for document analysis."""
        return {
            "title": "Concise Title for the Analysis",
            "analysis_sections": {
                "descriptive_section_name_1": {
                    "Analysis": "Detailed analysis text for this section...",
                    "Supporting_Phrases": [
                        "Exact quote 1 from the document text...",
                        "Exact quote 2, potentially longer..."
                        # NOTE: No chunk_id requested anymore
                    ],
                    "Context": "Optional context about this section (e.g., clause number)"
                },
                "another_descriptive_name": {
                    "Analysis": "Analysis text here...",
                    "Supporting_Phrases": [
                         "Exact quote 3...",
                         # Use "No relevant phrase found." if applicable
                    ],
                    "Context": "Optional context"
                }
                # Add more sections as identified by the AI
            }
        }

    async def analyze_document(self, document_text: str, filename: str, user_prompt: str) -> str:
        """Analyzes the document text based on the user prompt."""
        try:
            schema_str = json.dumps(self.output_schema_analysis, indent=2)

            system_prompt = f"""You are an intelligent document analyser specializing in legal and financial documents. Your task is to analyze the provided document text based on the user's prompt and provide structured output following a specific JSON schema.

### Core Instructions:
1.  **Analyze Thoroughly:** Read the user prompt and the document excerpts carefully. Perform the requested analysis.
2.  **Strict JSON Output:** Your entire response MUST be a single JSON object matching the schema provided below. Do not include any introductory text, explanations, apologies, or markdown formatting (` ```json`, ` ``` `) outside the JSON structure.
3.  **Descriptive Section Names:** Use lowercase snake_case for keys within `analysis_sections` (e.g., `cancellation_rights`, `liability_limitations`). These names should accurately reflect the content of the analysis in that section.
4.  **Exact Supporting Phrases:** The `Supporting_Phrases` array must contain *only direct, verbatim quotes* from the 'Document Excerpts'. Preserve original case, punctuation, and formatting within the quotes. Do *not* paraphrase or summarize. Aim for complete sentences or meaningful clauses.
5.  **No Phrase Found:** If no relevant phrase directly supports the analysis point for a section, include the exact string "No relevant phrase found." in the `Supporting_Phrases` array for that section.
6.  **Focus on Excerpts:** Base your analysis *only* on the text provided under '### Document Excerpts:'. Do not infer information not present.
7.  **Legal/Financial Context:** Pay attention to clause numbers, definitions, conditions, exceptions, and precise wording common in legal/financial texts.

### JSON Output Schema:
```json
{schema_str}
Use code with caution.
Python
"""  # End of multi-line f-string
            
            human_prompt = f"""Please analyze the following document based on the user prompt.

Document Name:
{filename}

User Prompt:
{user_prompt}

Document Excerpts:
{document_text}

Generate the analysis and supporting phrases strictly following the JSON schema provided in the system instructions. Ensure supporting phrases are exact quotes from the excerpts."""  # End of multi-line f-string

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ]

            logger.info(f"Sending analysis request for {filename} to AI.")
            # logger.debug(f"AI Analysis Request Messages: {json.dumps(messages, indent=2)}")

            response_content = await self._get_completion(messages)
            logger.info(f"Received AI analysis response for {filename}.")

            # Attempt to clean and parse the JSON
            try:
                # Basic cleaning: remove potential markdown fences and strip whitespace
                cleaned_response = response_content.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                # Validate if it's valid JSON
                parsed_json = json.loads(cleaned_response)

                # Optional: Validate against the schema structure (basic check)
                if "analysis_sections" not in parsed_json:
                     logger.error("AI response missing 'analysis_sections' key.")
                     raise ValueError("AI response missing 'analysis_sections' key.")
                if not isinstance(parsed_json["analysis_sections"], dict):
                     logger.error("'analysis_sections' in AI response is not a dictionary.")
                     raise ValueError("'analysis_sections' in AI response is not a dictionary.")

                logger.info("Successfully parsed AI analysis JSON response.")
                return json.dumps(parsed_json, indent=2) # Return the validated/parsed JSON string

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse AI analysis response as JSON: {json_err}")
                logger.error(f"Raw response content was: {response_content}")
                # Attempt to extract JSON using regex as a fallback
                match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if match:
                    logger.warning("Attempting to extract JSON using regex fallback.")
                    try:
                        extracted_json_str = match.group(0)
                        parsed_json = json.loads(extracted_json_str)
                        if "analysis_sections" in parsed_json: # Basic check again
                             logger.info("Successfully parsed AI analysis JSON using regex fallback.")
                             return json.dumps(parsed_json, indent=2)
                        else:
                             raise ValueError("Extracted JSON missing 'analysis_sections'.")
                    except Exception as fallback_err:
                        logger.error(f"Regex JSON extraction fallback also failed: {fallback_err}")
                        raise ValueError(f"AI response was not valid JSON and fallback extraction failed. Raw response: {response_content[:500]}...") from json_err
                else:
                     raise ValueError(f"AI response was not valid JSON and no JSON object found via regex. Raw response: {response_content[:500]}...") from json_err
            except ValueError as val_err:
                 # Catch schema validation errors or other ValueErrors
                 logger.error(f"Error validating AI response structure: {val_err}")
                 raise # Re-raise the validation error

        except Exception as e:
            logger.error(f"Error during AI document analysis for {filename}: {str(e)}", exc_info=True)
            # Return an error JSON structure
            error_response = {
                "title": f"Error Analyzing {filename}",
                "analysis_sections": {
                    "error": {
                        "Analysis": f"An error occurred during analysis: {str(e)}",
                        "Supporting_Phrases": ["No relevant phrase found."],
                        "Context": "System Error"
                    }
                }
            }
            return json.dumps(error_response, indent=2)

    async def generate_keywords(self, prompt: str) -> List[str]:
        """Generates relevant keywords based on the analysis prompt."""
        try:
            system_prompt = """You are an expert in extracting relevant keywords and phrases from financial and legal analysis prompts, particularly for Multilateral Development Banks (MDBs). Your goal is to generate a concise list of keywords that will be effective for searching related concepts in documents.
Use code with caution.
Instructions:

Analyze the user prompt to understand the core concepts and entities.

Generate a list of 5-15 highly relevant keywords or short phrases (2-3 words).

Include variations (e.g., "terminate", "termination").

Focus on nouns, verbs, and key adjectives/adverbs relevant to the prompt's intent.

Return only a JSON object containing a single key "keywords" which holds an array of strings. Do not include explanations or markdown.

Example:
User Prompt: "Confirm if the lender has the sole discretion to cancel the undisbursed loan commitment."
Your Output:

{
  "keywords": [
    "cancel",
    "cancellation",
    "terminate",
    "termination",
    "lender discretion",
    "sole discretion",
    "undisbursed",
    "commitment",
    "loan commitment",
    "right to cancel"
  ]
}
```"""  # End of multi-line string
            human_prompt = f"Generate keywords for the following prompt:\n\n{prompt}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ]

            logger.info(f"Sending keyword generation request for prompt: {prompt[:100]}...")
            response_content = await self._get_completion(messages, model="databricks-llama-3-8b-instruct") # Use a faster model if possible
            logger.info("Received keyword generation response.")

            try:
                # Clean and parse JSON
                cleaned_response = response_content.strip()
                if cleaned_response.startswith("```json"): cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"): cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                parsed_json = json.loads(cleaned_response)
                keywords = parsed_json.get("keywords", [])

                if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                    logger.info(f"Generated {len(keywords)} keywords.")
                    return keywords
                else:
                    logger.error(f"Invalid keyword format in response: {keywords}")
                    return []

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse keyword generation response as JSON: {json_err}")
                logger.error(f"Raw response content was: {response_content}")
                return []
            except Exception as parse_err:
                logger.error(f"Error processing keyword response: {parse_err}")
                return []

        except Exception as e:
            logger.error(f"Error during keyword generation: {str(e)}", exc_info=True)
            return []

    # --- Chat response method can be added here if needed, similar structure ---

# --- Document Processors ---

class PDFProcessor:
    """Handles PDF processing, chunking, verification, and annotation."""
    def __init__(self, pdf_bytes: bytes):
        if not isinstance(pdf_bytes, bytes):
             raise ValueError("pdf_bytes must be of type bytes")
        self.pdf_bytes = pdf_bytes
        self._chunks: List[Dict[str, Any]] = []
        self._full_text: Optional[str] = None
        logger.info(f"PDFProcessor initialized with {len(pdf_bytes)} bytes.")

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        if not self._chunks:
            self.extract_structured_text_and_chunks() # Lazy extraction
        return self._chunks

    @property
    def full_text(self) -> str:
         if self._full_text is None:
              self.extract_structured_text_and_chunks() # Lazy extraction
         return self._full_text if self._full_text is not None else ""


    def extract_structured_text_and_chunks(self) -> Tuple[List[Dict[str, Any]], str]:
        """Extracts text using PyMuPDF blocks and groups them into chunks."""
        if self._chunks: # Already processed
             return self._chunks, self._full_text if self._full_text is not None else ""

        self._chunks = []
        all_text_parts = []
        current_chunk_id = 0
        doc = None

        try:
            logger.info("Starting structured text extraction and chunking...")
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            logger.info(f"PDF opened with {doc.page_count} pages.")

            for page_num, page in enumerate(doc):
                try:
                    # Get text blocks with coordinates
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: (b[1], b[0])) # Sort by top coordinate, then left

                    current_chunk_text = []
                    current_chunk_bboxes = []
                    last_y1 = 0

                    for b in blocks:
                        x0, y0, x1, y1, text, block_no, block_type = b
                        block_text = text.strip()
                        if not block_text: # Skip empty blocks
                            continue

                        block_rect = fitz.Rect(x0, y0, x1, y1)

                        # --- Paragraph Grouping Logic ---
                        # Start a new chunk if:
                        # 1. It's the first block.
                        # 2. There's a significant vertical gap from the last block.
                        # 3. (Optional) Block type changes significantly (e.g., from text to image - less relevant here).
                        vertical_gap = y0 - last_y1
                        is_new_paragraph = not current_chunk_text or (vertical_gap > 5) # Adjust gap threshold as needed

                        if is_new_paragraph and current_chunk_text:
                            # Save the previous chunk
                            if len(normalize_text(" ".join(current_chunk_text))) > 10: # Only save substantial chunks
                                self._chunks.append({
                                    "chunk_id": f"chunk_{current_chunk_id}",
                                    "text": " ".join(current_chunk_text),
                                    "page_num": page_num,
                                    "bboxes": current_chunk_bboxes,
                                })
                                all_text_parts.append(" ".join(current_chunk_text))
                                current_chunk_id += 1
                            # Start new chunk
                            current_chunk_text = [block_text]
                            current_chunk_bboxes = [block_rect]
                        else:
                            # Add to the current chunk
                            current_chunk_text.append(block_text)
                            current_chunk_bboxes.append(block_rect)

                        last_y1 = y1

                    # Save the last chunk of the page
                    if current_chunk_text and len(normalize_text(" ".join(current_chunk_text))) > 10:
                        self._chunks.append({
                            "chunk_id": f"chunk_{current_chunk_id}",
                            "text": " ".join(current_chunk_text),
                            "page_num": page_num,
                            "bboxes": current_chunk_bboxes,
                        })
                        all_text_parts.append(" ".join(current_chunk_text))
                        current_chunk_id += 1

                except Exception as page_err:
                    logger.error(f"Error processing page {page_num}: {page_err}")

            self._full_text = "\n\n".join(all_text_parts)
            logger.info(f"Extraction complete. Generated {len(self._chunks)} chunks. Total text length: {len(self._full_text)} chars.")

        except Exception as e:
            logger.error(f"Failed to extract text/chunks: {str(e)}", exc_info=True)
            self._full_text = "" # Ensure empty string on failure
        finally:
            if doc:
                doc.close()

        return self._chunks, self._full_text if self._full_text is not None else ""

    def verify_and_locate_phrases(self, ai_analysis_json_str: str) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
        """Verifies AI phrases against chunks using fuzzy matching and locates them."""
        verification_results = {}
        phrase_locations = {}
        chunks_data = self.chunks # Ensure chunks are extracted

        if not chunks_data:
             logger.warning("No chunks available for verification.")
             return {}, {}

        try:
            ai_analysis = json.loads(ai_analysis_json_str)
            phrases_to_verify = []

            # Extract all supporting phrases from the AI analysis
            for section_data in ai_analysis.get("analysis_sections", {}).values():
                if isinstance(section_data, dict):
                    phrases = section_data.get("Supporting_Phrases", [])
                    if isinstance(phrases, list):
                         for phrase in phrases:
                              # Handle both dict and str format from AI (though should be str now)
                              p_text = ""
                              if isinstance(phrase, dict):
                                   p_text = phrase.get("text", "")
                              elif isinstance(phrase, str):
                                   p_text = phrase

                              if p_text and p_text != "No relevant phrase found.":
                                   phrases_to_verify.append(p_text.strip())

            logger.info(f"Starting verification for {len(phrases_to_verify)} unique phrases.")

            # Pre-normalize chunk texts
            normalized_chunks = [(chunk, normalize_text(chunk['text'])) for chunk in chunks_data]

            # --- Location Finding requires the document to be open ---
            doc = None
            try:
                doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")

                for original_phrase in phrases_to_verify:
                    verification_results[original_phrase] = False # Initialize
                    phrase_locations[original_phrase] = []
                    normalized_phrase = normalize_text(remove_markdown_formatting(original_phrase))

                    if not normalized_phrase: continue

                    found_match_for_phrase = False
                    best_score_for_phrase = 0

                    for chunk, norm_chunk_text in normalized_chunks:
                        if not norm_chunk_text: continue

                        # --- Fuzzy Match ---
                        score = fuzz.partial_ratio(normalized_phrase, norm_chunk_text)

                        if score >= FUZZY_MATCH_THRESHOLD:
                            if not found_match_for_phrase: # Log first verification
                                 logger.info(f"Verified (Score: {score}): '{original_phrase[:60]}...' potentially in chunk {chunk['chunk_id']}")
                            found_match_for_phrase = True
                            verification_results[original_phrase] = True
                            best_score_for_phrase = max(best_score_for_phrase, score)

                            # --- Precise Location Search using PyMuPDF ---
                            page_num = chunk['page_num']
                            if 0 <= page_num < doc.page_count:
                                page = doc[page_num]
                                # Use the union of bounding boxes as the clip area
                                clip_rect = fitz.Rect() # Empty rect
                                for bbox in chunk['bboxes']:
                                     clip_rect.include_rect(bbox)

                                if not clip_rect.is_empty:
                                     try:
                                          # 1. Clean the phrase before searching
                                          cleaned_search_phrase = remove_markdown_formatting(original_phrase)
                                          cleaned_search_phrase = re.sub(r'\s+', ' ', cleaned_search_phrase).strip() # Normalize whitespace

                                          # Search for the CLEANED ORIGINAL phrase within the chunk's area
                                          instances = page.search_for(cleaned_search_phrase, clip=clip_rect, quads=False)

                                          if instances:
                                              logger.debug(f"Found {len(instances)} instance(s) via search_for in chunk {chunk['chunk_id']} area for '{cleaned_search_phrase[:60]}...'")
                                              for rect in instances:
                                                  phrase_locations[original_phrase].append({
                                                      "page_num": page_num, # 0-indexed
                                                      "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                                                      "chunk_id": chunk['chunk_id'],
                                                      "match_score": score, # Store chunk score with exact location
                                                      "method": "exact_cleaned_search" # Updated method name
                                                  })
                                          else:
                                              # 2. Fallback to chunk bounding box if exact search fails after fuzzy chunk match
                                              logger.debug(f"Exact search failed for '{cleaned_search_phrase[:60]}...' in chunk {chunk['chunk_id']} area (score: {score}). Falling back to chunk bbox.")
                                              phrase_locations[original_phrase].append({
                                                  "page_num": page_num,
                                                  "rect": [clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1],
                                                  "chunk_id": chunk['chunk_id'],
                                                  "match_score": score, # Use the chunk score
                                                  "method": "fuzzy_chunk_fallback" # New method name
                                              })

                                     except Exception as search_err:
                                          logger.error(f"Error during search_for or fallback in chunk {chunk['chunk_id']}: {search_err}")
                                else:
                                      logger.warning(f"Chunk {chunk['chunk_id']} had empty bounding box, skipping search_for.")
                            else:
                                 logger.warning(f"Invalid page number {page_num} for chunk {chunk['chunk_id']}.")

                    if not found_match_for_phrase:
                        logger.warning(f"NOT Verified: '{original_phrase[:60]}...' did not meet fuzzy threshold ({FUZZY_MATCH_THRESHOLD}) in any chunk.")

            finally:
                 if doc:
                      doc.close()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI analysis JSON for verification: {e}")
        except Exception as e:
            logger.error(f"Error during phrase verification and location: {str(e)}", exc_info=True)

        return verification_results, phrase_locations

    def add_annotations(self, phrase_locations: Dict[str, List[Dict[str, Any]]]) -> bytes:
        """Adds highlights to the PDF based on found phrase locations."""
        if not phrase_locations:
            logger.warning("No phrase locations provided for annotation. Returning original PDF bytes.")
            return self.pdf_bytes

        doc = None
        try:
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            annotated_count = 0
            highlight_color = [1, 0.9, 0.3] # Yellow
            fallback_color = [0.5, 0.7, 1.0] # Blue for fallback/chunk bbox

            for phrase, locations in phrase_locations.items():
                for loc in locations:
                    try:
                        page_num = loc['page_num']
                        rect_coords = loc['rect']
                        method = loc.get('method', 'unknown')

                        if 0 <= page_num < doc.page_count:
                            page = doc[page_num]
                            rect = fitz.Rect(rect_coords)
                            if not rect.is_empty:
                                # Use fallback color if the method indicates a less precise location
                                color = fallback_color if method in ["fuzzy_chunk_fallback", "fuzzy_chunk_bbox"] else highlight_color
                                highlight = page.add_highlight_annot(rect)
                                highlight.set_colors(stroke=color)
                                highlight.set_info(content=f"Verified ({method}): {phrase[:100]}...")
                                highlight.update(opacity=0.4)
                                annotated_count += 1
                            else:
                                 logger.warning(f"Skipping annotation for empty rectangle for phrase '{phrase[:50]}...' on page {page_num}.")
                        else:
                            logger.warning(f"Invalid page number {page_num} for annotation.")
                    except Exception as annot_err:
                        logger.error(f"Error adding annotation for phrase '{phrase[:50]}...' at {loc}: {annot_err}")

            if annotated_count > 0:
                 logger.info(f"Added {annotated_count} highlight annotations.")
                 output_buffer = BytesIO()
                 doc.save(output_buffer, garbage=4, deflate=True)
                 annotated_bytes = output_buffer.getvalue()
            else:
                 logger.warning("No annotations were successfully added. Returning original PDF bytes.")
                 annotated_bytes = self.pdf_bytes

            return annotated_bytes

        except Exception as e:
            logger.error(f"Failed to add annotations: {str(e)}", exc_info=True)
            return self.pdf_bytes # Return original on error
        finally:
            if doc:
                doc.close()


class WordProcessor:
    """Handles Word document conversion to PDF."""
    def __init__(self, docx_bytes: bytes):
        if not isinstance(docx_bytes, bytes):
             raise ValueError("docx_bytes must be of type bytes")
        self.docx_bytes = docx_bytes
        logger.info(f"WordProcessor initialized with {len(docx_bytes)} bytes.")

    def convert_to_pdf_bytes(self) -> Optional[bytes]:
        """Converts the DOCX to PDF using a basic text-dumping approach."""
        logger.warning("Using basic DOCX to PDF conversion (text dump). Formatting will be lost.")
        try:
            # 1. Extract text using python-docx
            doc = DocxDocument(BytesIO(self.docx_bytes))
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            full_text.append(para.text)
            extracted_text = "\n".join(full_text)

            if not extracted_text.strip():
                 logger.warning("No text extracted from DOCX file.")
                 return self._create_minimal_empty_pdf() # Return empty PDF

            # 2. Create basic PDF using fitz
            pdf_doc = fitz.open()
            page = pdf_doc.new_page()
            rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
            res = page.insert_textbox(rect, extracted_text, fontsize=11, align=fitz.TEXT_ALIGN_LEFT)
            if res < 0:
                logger.warning("Text might be truncated during basic PDF creation.")

            output_buffer = BytesIO()
            pdf_doc.save(output_buffer)
            pdf_doc.close()
            logger.info("Successfully created basic PDF from DOCX text.")
            return output_buffer.getvalue()

        except ImportError:
             logger.error("python-docx not installed. Cannot process Word files.")
             raise Exception("python-docx is required to process Word documents.") # Re-raise
        except Exception as e:
            logger.error(f"Error during basic DOCX to PDF conversion: {e}", exc_info=True)
            return self._create_minimal_empty_pdf() # Return empty PDF on error

    def _create_minimal_empty_pdf(self) -> bytes:
         """Creates a minimal valid empty PDF."""
         try:
              min_pdf = fitz.open()
              min_pdf.new_page()
              buffer = BytesIO()
              min_pdf.save(buffer)
              min_pdf.close()
              logger.warning("Created minimal empty PDF as fallback.")
              return buffer.getvalue()
         except Exception as min_e:
              logger.error(f"Failed to create minimal PDF: {min_e}")
              # Absolute last resort
              return b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n109\n%%EOF\n"


# --- Streamlit UI Functions (Adapted from user's files) ---

# --- PDF Viewer Logic (Adapted from pdf_viewer.py) ---
def update_pdf_view(pdf_bytes=None, page_num=None, filename=None):
    """Updates session state for the PDF viewer."""
    if not isinstance(page_num, int) or page_num < 1:
        page_num = 1
    if pdf_bytes is None:
        pdf_bytes = st.session_state.get('pdf_bytes')
    if filename is None:
        filename = st.session_state.get('current_pdf_name', 'document.pdf')

    st.session_state.show_pdf = True
    st.session_state.pdf_page = page_num
    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.current_pdf_name = filename
    logger.info(f"Updating PDF view to page {page_num} of {filename}")
    # Note: No st.rerun() here, rely on Streamlit's natural flow

def display_pdf_viewer():
    """Renders the PDF viewer based on session state."""
    pdf_bytes = st.session_state.get('pdf_bytes')
    show_pdf = st.session_state.get('show_pdf', False)
    current_page = st.session_state.get('pdf_page', 1)
    filename = st.session_state.get('current_pdf_name', 'PDF Viewer')

    with st.expander("📄 PDF Viewer", expanded=show_pdf):
        if not show_pdf or not pdf_bytes:
            st.info("No PDF selected or available. Process a file or click a citation.")
            return

        tmp_file_path = None
        pdf_doc = None # pdfplumber doc
        fitz_doc = None # fitz doc for page count

        try:
            # Get total pages using fitz (more reliable)
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = fitz_doc.page_count
            if total_pages == 0:
                 st.warning("The PDF document appears to have 0 pages.")
                 return

            # Navigation
            st.markdown(f"**{filename}**")
            cols = st.columns([1, 3, 1])
            with cols[0]:
                if st.button("⬅️", key="pdf_prev", disabled=(current_page <= 1)):
                    st.session_state.pdf_page -= 1
                    st.rerun() # Rerun to update view
            with cols[1]:
                 # Ensure slider/input uses valid range
                nav_key = f"nav_{filename}_{current_page}" # More unique key
                selected_page = st.number_input(
                     "Page",
                     min_value=1,
                     max_value=total_pages,
                     value=current_page,
                     step=1,
                     key=nav_key,
                     label_visibility="collapsed"
                 )
                if selected_page != current_page:
                     st.session_state.pdf_page = selected_page
                     st.rerun() # Rerun to update view
            with cols[2]:
                if st.button("➡️", key="pdf_next", disabled=(current_page >= total_pages)):
                    st.session_state.pdf_page += 1
                    st.rerun() # Rerun to update view

            st.caption(f"Page {current_page} of {total_pages}")

            # Render page using pdfplumber (as in original code)
            # Create temp file for pdfplumber
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_file_path = tmp_file.name

            pdf_doc = pdfplumber.open(tmp_file_path)
            if 0 <= current_page - 1 < len(pdf_doc.pages):
                 page = pdf_doc.pages[current_page - 1]
                 img = page.to_image(resolution=150).original # Lower res for faster load? Adjust if needed
                 st.image(img, use_container_width=True)
            else:
                 st.error(f"Cannot display page {current_page}. Page index out of range.")


        except Exception as e:
            logger.error(f"Error displaying PDF viewer: {e}", exc_info=True)
            st.error(f"Error displaying PDF: {e}")
        finally:
            if pdf_doc: pdf_doc.close()
            if fitz_doc: fitz_doc.close()
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    logger.error(f"Error deleting temp PDF file {tmp_file_path}: {e}")
                    atexit.register(os.unlink, tmp_file_path) # Ensure cleanup on exit

# --- Analysis Display Logic (Adapted from analysis_display.py) ---
def find_best_location(locations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the best location (prioritizing exact matches, then highest score, then earliest page)."""
    if not locations:
        return None

    def sort_key(loc):
        method = loc.get('method', 'unknown')
        score = loc.get('match_score', 0)
        page_num = loc.get('page_num', 9999)

        # Assign a priority score based on method (lower is better)
        method_priority = 0 if method == 'exact_cleaned_search' else 1

        # Sort by: method priority (ascending), score (descending), page number (ascending)
        return (method_priority, -score, page_num)

    # Sort locations using the custom key
    locations.sort(key=sort_key)
    return locations[0] # Return the best one after sorting

def display_analysis_results(analysis_results: List[Dict[str, Any]]):
    """Displays the analysis sections and citations."""
    if not analysis_results:
        logger.info("No analysis results to display.")
        return

    # Create columns for analysis and PDF viewer
    analysis_col, pdf_col = st.columns([2.5, 1.5], gap="medium")

    with analysis_col:
        st.markdown("### AI Analysis Results")
        for i, result in enumerate(analysis_results):
            is_first_result = (i == 0)
            filename = result.get("filename", "Unknown File")
            ai_analysis_json_str = result.get("ai_analysis", "{}")
            verification_results = result.get("verification_results", {})
            phrase_locations = result.get("phrase_locations", {})
            annotated_pdf_b64 = result.get("annotated_pdf") # Base64 encoded

            st.markdown(f"#### {filename}")

            try:
                ai_analysis = json.loads(ai_analysis_json_str)
                analysis_sections = ai_analysis.get("analysis_sections", {})

                if not analysis_sections:
                     st.warning("No analysis sections found in the AI response.")
                     continue

                # Display title if present
                if ai_analysis.get("title"):
                     st.markdown(f"**{ai_analysis['title']}**")

                citation_counter = 0 # Reset for each file/result
                for section_name, section_data in analysis_sections.items():
                    display_section_name = section_name.replace('_', ' ').title()
                    with st.container(border=True):
                         st.subheader(display_section_name)

                         if isinstance(section_data, dict):
                              # Analysis Text
                              if section_data.get("Analysis"):
                                   st.write(section_data["Analysis"])
                              # Context Text
                              if section_data.get("Context"):
                                   st.caption(f"Context: {section_data['Context']}")

                              # Supporting Phrases / Citations
                              supporting_phrases = section_data.get("Supporting_Phrases", [])
                              if supporting_phrases:
                                   with st.expander("Supporting Citations", expanded=False):
                                       for phrase_text in supporting_phrases:
                                           if not isinstance(phrase_text, str) or phrase_text == "No relevant phrase found.":
                                                # Handle "No phrase found" case if needed, or just skip
                                                # st.caption("No supporting citation provided by AI.")
                                                continue

                                           citation_counter += 1
                                           is_verified = verification_results.get(phrase_text, False)
                                           locations = phrase_locations.get(phrase_text, [])
                                           best_location = find_best_location(locations)

                                           # Status Icon
                                           status_emoji = "✅" if is_verified else "❓"
                                           score_info = f"(Score: {best_location['match_score']:.0f})" if best_location and 'match_score' in best_location else ""
                                           method_info = f"Method: {best_location['method']}" if best_location and 'method' in best_location else ""

                                           # Display Citation
                                           st.markdown(f"""
                                           <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px; margin-bottom: 10px; background-color: #f9f9f9;">
                                               <div style="margin-bottom: 5px;">
                                                   <span style="font-weight: bold;">Citation {citation_counter}</span> {status_emoji}
                                                   <span style="font-size: 0.8em; color: #666;">{score_info} {method_info}</span>
                                               </div>
                                               <div style="color: #424242; line-height: 1.4;">
                                                   "{phrase_text}"
                                               </div>
                                           </div>
                                           """, unsafe_allow_html=True)

                                           # Go to Page Button
                                           if is_verified and best_location and 'page_num' in best_location:
                                                page_num_0_indexed = best_location['page_num']
                                                page_num_1_indexed = page_num_0_indexed + 1
                                                button_key = f"goto_{filename}_{citation_counter}"
                                                button_label = f"Go to Page {page_num_1_indexed}"
                                                if filename != st.session_state.get('current_pdf_name'):
                                                     button_label += f" (in {filename})"

                                                # Use columns to right-align button
                                                _, btn_col = st.columns([0.7, 0.3])
                                                with btn_col:
                                                    if st.button(button_label, key=button_key, type="secondary"):
                                                        if annotated_pdf_b64:
                                                             pdf_bytes = base64.b64decode(annotated_pdf_b64)
                                                             # Update view expects 1-based page num
                                                             update_pdf_view(pdf_bytes=pdf_bytes, page_num=page_num_1_indexed, filename=filename)
                                                             st.rerun() # Rerun to update PDF viewer section
                                                        else:
                                                             st.warning("Annotated PDF data not found.")
                                           elif is_verified:
                                                _, btn_col = st.columns([0.7, 0.3])
                                                with btn_col:
                                                     st.caption("Location not found")
                         else:
                              st.warning(f"Unexpected data format for section '{section_name}'.")

            except json.JSONDecodeError:
                st.error(f"Failed to decode AI analysis JSON for {filename}. Raw data:")
                st.code(ai_analysis_json_str)
            except Exception as display_err:
                 logger.error(f"Error displaying analysis for {filename}: {display_err}", exc_info=True)
                 st.error(f"Error displaying analysis results for {filename}: {display_err}")

    with pdf_col:
         # --- Analysis Tools & PDF Viewer Section ---
         # st.write("---") # Separator
         st.markdown("### Analysis Tools")

         # --- Chat Interface Expander ---
         with st.expander("💬 SmartChat (Beta)", expanded=False):
             # TODO: Implement or adapt chat interface logic here if needed
             st.info("Chat feature placeholder.")
             # chat_interface() # Call your chat function if available

         # --- Export Expander ---
         with st.expander("📊 Export Results", expanded=False):
             if analysis_results:
                  col1, col2 = st.columns(2)
                  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                  try:
                       # Flatten results for export
                       flat_data = []
                       for res in analysis_results:
                           fname = res.get('filename', 'N/A')
                           try: ai_data = json.loads(res.get('ai_analysis', '{}'))
                           except: ai_data = {}
                           title = ai_data.get('title', '')
                           for sec_name, sec_data in ai_data.get('analysis_sections', {}).items():
                                analysis = sec_data.get('Analysis', '')
                                context = sec_data.get('Context', '')
                                phrases = sec_data.get('Supporting_Phrases', [])
                                for phrase in phrases:
                                     verified = res.get('verification_results', {}).get(phrase, False)
                                     locs = res.get('phrase_locations', {}).get(phrase, [])
                                     best_loc = find_best_location(locs)
                                     page = best_loc['page_num'] + 1 if best_loc else 'N/A'
                                     score = best_loc['match_score'] if best_loc else 'N/A'
                                     flat_data.append({
                                          'Filename': fname, 'AI Title': title, 'Section': sec_name,
                                          'Analysis': analysis, 'Context': context, 'Supporting Phrase': phrase,
                                          'Verified': verified, 'Page': page, 'Match Score': score
                                     })

                       df = pd.DataFrame(flat_data)
                       excel_buffer = BytesIO()
                       df.to_excel(excel_buffer, index=False, engine='openpyxl')
                       excel_buffer.seek(0)
                       with col1:
                           st.download_button("📥 Export Excel", excel_buffer, f"analysis_{timestamp}.xlsx",
                                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                               key="export_excel_main")

                       # Word Export (Simplified - Just dump JSON for now)
                       word_buffer = BytesIO()
                       # TODO: Implement better Word export if needed
                       word_buffer.write(json.dumps(analysis_results, indent=2).encode('utf-8'))
                       word_buffer.seek(0)
                       with col2:
                           st.download_button("📄 Export Word (JSON)", word_buffer, f"analysis_{timestamp}_raw.txt",
                                              "text/plain", key="export_word_main")

                  except Exception as export_err:
                       logger.error(f"Export failed: {export_err}", exc_info=True)
                       st.error(f"Export failed: {export_err}")
             else:
                  st.info("No results to export.")

         # --- Report Issue Expander ---
         with st.expander("⚠️ Report Issue", expanded=False):
              st.info("Issue reporting placeholder.")
              # Add issue reporting UI and logic here

         # --- PDF Viewer Display ---
         display_pdf_viewer()


# --- Main Application Logic (Adapted from app.py) ---

def process_file_wrapper(args):
    """Wrapper for process_file to handle args tuple for thread pool."""
    uploaded_file_data, filename, keywords_str, threshold, user_prompt, use_keywords, use_advanced_extraction = args

    # Recreate file-like object for processing functions
    uploaded_file_obj = BytesIO(uploaded_file_data)
    uploaded_file_obj.name = filename # Add name attribute expected by downstream funcs

    logger.info(f"Thread {threading.current_thread().name} processing: {filename}")
    try:
        # Set thread-local variables instead of using session state directly
        # This removes the dependency on session state in the worker threads
        thread_use_keywords = use_keywords
        thread_use_advanced = use_advanced_extraction

        file_extension = Path(filename).suffix.lower()

        if file_extension == '.pdf':
            processor = PDFProcessor(uploaded_file_data)
            chunks, full_text = processor.extract_structured_text_and_chunks()

            # --- AI Analysis ---
            analyzer = DocumentAnalyzer() # Assumes lazy init works per thread
            ai_analysis_json_str = run_async(analyzer.analyze_document(full_text, filename, user_prompt))

            # --- Verification & Location ---
            verification_results, phrase_locations = processor.verify_and_locate_phrases(ai_analysis_json_str)

            # --- Annotation ---
            annotated_pdf_bytes = processor.add_annotations(phrase_locations)

            return {
                "filename": filename,
                "annotated_pdf": base64.b64encode(annotated_pdf_bytes).decode(),
                "verification_results": verification_results,
                "phrase_locations": phrase_locations,
                "ai_analysis": ai_analysis_json_str,
                # "chunks": chunks, # Avoid sending large chunks back if not needed for display
                # "extracted_text": full_text # Avoid sending full text back
            }

        elif file_extension == '.docx':
            word_processor = WordProcessor(uploaded_file_data)
            pdf_bytes = word_processor.convert_to_pdf_bytes()
            if not pdf_bytes:
                raise ValueError("Failed to convert DOCX to PDF.")

            # Now process the converted PDF
            processor = PDFProcessor(pdf_bytes)
            chunks, full_text = processor.extract_structured_text_and_chunks()

            # --- AI Analysis ---
            analyzer = DocumentAnalyzer()
            ai_analysis_json_str = run_async(analyzer.analyze_document(full_text, filename, user_prompt))

            # --- Verification & Location ---
            verification_results, phrase_locations = processor.verify_and_locate_phrases(ai_analysis_json_str)

            # --- Annotation ---
            # Annotate the *converted* PDF
            annotated_pdf_bytes = processor.add_annotations(phrase_locations)

            return {
                "filename": filename,
                "annotated_pdf": base64.b64encode(annotated_pdf_bytes).decode(), # The annotated *converted* PDF
                "verification_results": verification_results,
                "phrase_locations": phrase_locations,
                "ai_analysis": ai_analysis_json_str,
            }
        else:
            logger.error(f"Unsupported file type skipped in wrapper: {filename}")
            return None

    except Exception as e:
        logger.error(f"Error in process_file_wrapper for {filename}: {str(e)}", exc_info=True)
        # Return error information structure
        return {
             "filename": filename,
             "error": str(e),
             "annotated_pdf": None,
             "verification_results": {},
             "phrase_locations": {},
             "ai_analysis": json.dumps({"error": f"Failed to process: {e}"})
        }


def display_page():
    """Main function to display the Streamlit page."""
    # --- Initialize Session State ---
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = []
    if 'show_pdf' not in st.session_state: st.session_state.show_pdf = False
    if 'pdf_page' not in st.session_state: st.session_state.pdf_page = 1
    if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
    if 'current_pdf_name' not in st.session_state: st.session_state.current_pdf_name = None
    if 'keywords_input' not in st.session_state: st.session_state.keywords_input = ""
    if 'user_prompt' not in st.session_state: st.session_state.user_prompt = ""
    if 'use_keywords' not in st.session_state: st.session_state.use_keywords = True # Default to keyword mode
    if 'use_advanced_extraction' not in st.session_state: st.session_state.use_advanced_extraction = False

    # Add other necessary initializations from your app.py if needed

    # --- UI Layout ---
    # Logos can be added here using get_base64_encoded_image and st.markdown
    st.markdown("<h1 style='text-align: center;'>SmartDocs Document Analysis</h1>", unsafe_allow_html=True)

    # --- File Upload ---
    uploaded_files = st.file_uploader(
        "Upload PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader" # Keep key consistent
    )

    if uploaded_files:
         # Store uploaded file references if they change
         if "processed_filenames" not in st.session_state or set(f.name for f in uploaded_files) != st.session_state.processed_filenames:
              st.session_state.uploaded_file_refs = uploaded_files
              st.session_state.processed_filenames = set(f.name for f in uploaded_files)
              # Optionally clear previous results when files change significantly
              # st.session_state.analysis_results = []
              # st.session_state.show_pdf = False

    # --- Analysis Inputs ---
    with st.container(border=True):
        st.subheader("Analysis Configuration")
        st.text_area(
            "Analysis Prompt",
            placeholder="Enter specific instructions for the analysis...",
            height=150,
            key="prompt_input_main",
            # Use session state directly for the prompt value
            value=st.session_state.get('user_prompt', ''),
            # Update session state when the text area changes
            on_change=lambda: setattr(st.session_state, 'user_prompt', st.session_state.get('prompt_input_main', ''))
        )

        # Use toggle widget for keywords filter
        use_keywords_toggle = st.toggle(
             "Use Keywords Filter (Optional)",
             value=st.session_state.get('use_keywords', True), # Read default from session state 
             help="If enabled, you can provide keywords. If disabled, analysis uses semantic relevance to the prompt.",
             key="use_keywords_toggle_main", # Keep key consistent
             # This will update the session state value behind the scenes
        )
        # Store the current toggle value in session state
        st.session_state['use_keywords'] = use_keywords_toggle

        # Initialize keywords_str to empty string. Its value will be determined by the input widget
        # or remain empty if the toggle is off. This variable is now primarily for passing to the processing function.
        keywords_str_for_processing = ""

        if st.session_state.get('use_keywords', True):
            # Display the text input for keywords
            st.text_input(
                "Keywords (comma-separated)",
                # Ensure value comes from session state for persistence across reruns
                value=st.session_state.get('keywords_input', ''),
                key="keywords_input_widget", # Use a distinct key for the widget
                help="Keywords to focus on if 'Use Keywords Filter' is enabled.",
                # Update session state when input changes
                on_change=lambda: setattr(st.session_state, 'keywords_input', st.session_state.get('keywords_input_widget', ''))
            )
            # Read the latest value from session state for processing
            keywords_str_for_processing = st.session_state.get('keywords_input', '')

            # Auto-generate button logic
            if st.button("🔄 Auto-generate keywords", disabled=not st.session_state.get('user_prompt', '')):
                 if st.session_state.get('user_prompt', ''):
                     with st.spinner("Generating keywords..."):
                         analyzer = DocumentAnalyzer() # Assuming DocumentAnalyzer is defined
                         gen_keywords = run_async(analyzer.generate_keywords(st.session_state.get('user_prompt', ''))) # Assuming run_async is defined
                         if gen_keywords:
                             # Update session state, which will update the text_input value on rerun
                             st.session_state.keywords_input = ", ".join(gen_keywords)
                             st.rerun() # Rerun to show the updated keywords in the input box
                         else:
                             st.warning("Could not auto-generate keywords.")
                 else:
                     st.warning("Please enter an analysis prompt to generate keywords.")
        else:
             # If not using keywords, ensure the session state value is cleared
             # (useful if the user toggles it off after entering keywords)
             if st.session_state.get('keywords_input'):
                  st.session_state.keywords_input = ""
             # keywords_str_for_processing remains ""

        # Read the current value of the advanced extraction toggle from session state
        use_advanced = st.session_state.get('use_advanced_extraction', False)
        st.toggle(
            "Use Advanced PDF Extraction",
            value=use_advanced,
            help="For PDFs only. Uses layout-aware extraction (slower, potentially better for complex layouts).",
            key="advanced_extraction_toggle_widget", # Distinct key
            # Update session state when the toggle changes
            on_change=lambda: setattr(st.session_state, 'use_advanced_extraction', st.session_state.get('advanced_extraction_toggle_widget', False))
        )
        # Update the variable used for processing based on current session state
        use_advanced_for_processing = st.session_state.get('use_advanced_extraction', False)


    # --- Process Button ---
    if st.button("Process Documents", type="primary", use_container_width=True):
        # Read necessary values from session state at the time of the button click
        files_to_process = st.session_state.get('uploaded_file_refs', [])
        current_user_prompt = st.session_state.get('user_prompt', '')
        current_use_keywords = st.session_state.get('use_keywords', True)
        # Use the keywords string that was determined based on the toggle state above
        current_keywords_str = keywords_str_for_processing
        # Read advanced extraction state for processing
        current_use_advanced = st.session_state.get('use_advanced_extraction', False)


        # --- Input Validation ---
        if not files_to_process:
            st.warning("Please upload one or more documents.")
        elif not current_user_prompt:
            st.error("Please enter an Analysis Prompt.")
        # Optional: Add back keyword validation if needed
        # elif current_use_keywords and not current_keywords_str.strip():
        #     st.warning("Keywords filter is enabled, but no keywords were provided.")
        else:
            # --- Start Processing ---
            st.session_state.analysis_results = [] # Clear previous results
            st.session_state.show_pdf = False      # Hide PDF viewer initially
            total_files = len(files_to_process)
            overall_start_time = datetime.now()
            # Preallocate results list to maintain order
            results_list = [None] * total_files

            # Prepare arguments for parallel processing using values captured at button click
            process_args = []
            valid_files_found = True
            for uploaded_file in files_to_process:
                try:
                    if hasattr(uploaded_file, 'getvalue'):
                         file_data = uploaded_file.getvalue()
                         process_args.append((
                              file_data,
                              uploaded_file.name,
                              current_keywords_str,     # Use value captured on click
                              FUZZY_MATCH_THRESHOLD,    # Use constant
                              current_user_prompt,      # Use value captured on click
                              current_use_keywords,     # Use value captured on click
                              current_use_advanced      # Use value captured on click
                         ))
                    else:
                         logger.error(f"Uploaded file object for {uploaded_file.name} is invalid or closed.")
                         st.error(f"File {uploaded_file.name} could not be read. Please re-upload.")
                         valid_files_found = False
                         break
                except Exception as read_err:
                    logger.error(f"Failed to read file {uploaded_file.name}: {read_err}", exc_info=True)
                    st.error(f"Failed to read file {uploaded_file.name}. Please re-upload.")
                    valid_files_found = False
                    break

            # Only proceed if files were read and args prepared successfully
            if valid_files_found and process_args:
                # --- Use st.spinner for simpler progress indication ---
                with st.spinner(f"Processing {total_files} documents..."):
                    # Define the function to run in threads (NO st calls inside)
                    def run_process_task(item_index: int, args_tuple: tuple):
                        """Processes a single file in a background thread."""
                        filename = args_tuple[1]
                        logger.info(f"Thread {threading.current_thread().name} starting: {filename}")
                        try:
                            result = process_file_wrapper(args_tuple) # Assumes process_file_wrapper is defined elsewhere
                            logger.info(f"Thread {threading.current_thread().name} finished: {filename}")
                            return item_index, result
                        except Exception as thread_err:
                            logger.error(f"Unhandled error in thread task for {filename}: {thread_err}", exc_info=True)
                            return item_index, {"filename": filename, "error": f"Unhandled thread error: {thread_err}"}

                    # --- Execute Tasks (Parallel or Sequential) ---
                    if ENABLE_PARALLEL and total_files > 1: # Assumes ENABLE_PARALLEL is defined
                        logger.info(f"Using ThreadPoolExecutor with {MAX_WORKERS} workers.") # Assumes MAX_WORKERS is defined
                        try:
                             with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                future_to_index = {
                                    executor.submit(run_process_task, i, arg_tuple): i
                                    for i, arg_tuple in enumerate(process_args)
                                }
                                for future in concurrent.futures.as_completed(future_to_index):
                                    original_index = future_to_index[future]
                                    try:
                                        _, result_data = future.result()
                                        results_list[original_index] = result_data
                                    except Exception as exc:
                                        logger.error(f'Thread task execution resulted in an exception: {exc}', exc_info=True)
                                        filename_for_error = process_args[original_index][1]
                                        results_list[original_index] = {"filename": filename_for_error, "error": f"Task execution failed: {exc}"}
                        except Exception as exec_err:
                             logger.error(f"ThreadPoolExecutor failed: {exec_err}", exc_info=True)
                             st.error(f"Error during parallel processing setup: {exec_err}. Try disabling parallel mode if issues persist.")
                             process_args = [] # Prevent proceeding

                    else: # Sequential execution
                        logger.info("Processing sequentially (Parallel disabled or only 1 file).")
                        for i, arg_tuple in enumerate(process_args):
                            _, result_data = run_process_task(i, arg_tuple)
                            results_list[i] = result_data

                    # --- Processing Done - Update State in Main Thread ---
                    st.session_state.analysis_results = [r for r in results_list if r is not None]

                    total_time = (datetime.now() - overall_start_time).total_seconds()
                    success_count = len([r for r in st.session_state.analysis_results if "error" not in r])
                    logger.info(f"Processing complete. Processed {success_count}/{total_files} files in {total_time:.2f}s.")

                    # --- Set initial PDF view ---
                    # Check if results exist and the first one doesn't contain an error key
                    if st.session_state.analysis_results and "error" not in st.session_state.analysis_results[0]:
                        first_result = st.session_state.analysis_results[0]
                        if first_result.get("annotated_pdf"):
                            try:
                                st.session_state.pdf_bytes = base64.b64decode(first_result["annotated_pdf"])
                                st.session_state.current_pdf_name = first_result.get("filename", "Unknown")
                                st.session_state.show_pdf = True
                                st.session_state.pdf_page = 1
                            except Exception as decode_err:
                                logger.error(f"Failed to decode/set initial PDF: {decode_err}", exc_info=True)
                                st.error("Failed to load initial PDF view.")
                        else:
                             logger.warning("First successful result did not contain annotated PDF data.")
                             st.warning("Processing complete, but couldn't display the first annotated document.")
                    elif st.session_state.analysis_results and "error" in st.session_state.analysis_results[0]:
                         logger.warning("First file processing resulted in an error. No initial PDF view shown.")
                         st.warning("First file failed processing. See errors below.")
                    else:
                         logger.warning("No results generated after processing.")

            # Force a rerun AFTER processing is complete and state is updated
            st.rerun()

            # --- UI Update after Spinner ---
            if 'success_count' in locals() and total_files > 0: # Check if processing actually ran
                 if success_count == total_files:
                      st.success(f"Successfully processed {total_files} files in {total_time:.2f} seconds.")
                 elif success_count > 0:
                      st.warning(f"Processing complete. Processed {success_count}/{total_files} files successfully in {total_time:.2f} seconds. Some files failed.")
                 else:
                      st.error(f"Processing failed for all {total_files} files in {total_time:.2f} seconds.")

# MOVED: Display Results (Outside and After the button block)
# This will run on every Streamlit rerun, showing results from session state
if st.session_state.get('analysis_results'):
     # Handle displaying results with errors
     errors = [r for r in st.session_state.get('analysis_results') if "error" in r]
     success_results = [r for r in st.session_state.get('analysis_results') if "error" not in r]

     if errors:
          st.error("Some files failed to process:")
          for error_res in errors:
               st.write(f"- **{error_res['filename']}**: {error_res['error']}")
          st.write("---")

     if success_results:
         # This function should create the columns and render the display
         display_analysis_results(success_results)
     elif not errors:
          st.warning("Processing finished, but no results were generated.")

# --- Main Execution ---
if __name__ == "__main__":
    display_page()