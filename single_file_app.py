# single_file_app.py

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import re
import tempfile
import threading
import queue
import atexit
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np
import pandas as pd
import pdfplumber  # For PDF viewer rendering
import streamlit as st
import torch  # Usually implicitly required by sentence-transformers
from docx import Document as DocxDocument  # Renamed to avoid conflict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz

# --- Configuration ---
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

# ****** SET PAGE CONFIG HERE (First Streamlit command) ******
st.set_page_config(layout="wide", page_title="SmartDocs Analysis")
# ****** END SET PAGE CONFIG ******

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
ENABLE_PARALLEL = os.environ.get("ENABLE_PARALLEL", "true").lower() == "true"
FUZZY_MATCH_THRESHOLD = 88  # Adjust this threshold (0-100)
RAG_TOP_K = 20  # Number of relevant chunks to retrieve
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller, faster model for embeddings

# --- Load Embedding Model (Load once globally) ---
embedding_model = None
try:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # Check for CUDA availability, fallback to CPU if needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    logger.info(f"Embedding model loaded successfully on device: {device}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}", exc_info=True)
    st.error(
        f"Fatal Error: Could not load embedding model '{EMBEDDING_MODEL_NAME}'. "
        "Please check installation and dependencies."
    )
    # Potentially exit or disable processing if model is critical
    # st.stop()


# --- Helper Functions ---

def normalize_text(text: Optional[str]) -> str:
    """Normalize text for comparison: lowercase, strip, whitespace."""
    if not text:
        return ""
    text = str(text)
    text = text.lower()  # Case-insensitive matching
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    # Optional: Remove simple punctuation if causing issues - use cautiously
    # text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def remove_markdown_formatting(text: Optional[str]) -> str:
    """Removes common markdown formatting."""
    if not text:
        return ""
    text = str(text)
    # Basic bold, italics, code
    text = re.sub(r"\*(\*|_)(.*?)\1\*?", r"\2", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Basic headings, blockquotes, lists
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
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

# --- RAG Retrieval Function ---
def retrieve_relevant_chunks(
    prompt: str,
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int,
) -> Tuple[str, List[str]]:
    """
    Retrieves the top_k most relevant chunks based on semantic similarity to the prompt.

    Args:
        prompt: The user's analysis prompt.
        chunks: A list of dictionaries, where each dict represents a chunk and has a 'text' key.
        model: The loaded SentenceTransformer model.
        top_k: The number of top chunks to retrieve.

    Returns:
        A tuple containing:
        - A single string concatenating the text of the top_k relevant chunks.
        - A list of chunk_ids for the retrieved chunks.
    """
    if not chunks or not prompt or model is None:
        logger.warning("RAG retrieval skipped: No chunks, prompt, or model available.")
        return "", []

    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    if not any(chunk_texts):
        logger.warning("RAG retrieval skipped: All chunk texts are empty.")
        return "", []

    try:
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks and prompt...")
        # Generate embeddings
        prompt_embedding = model.encode(
            prompt, convert_to_tensor=True, show_progress_bar=False
        )
        # Filter out empty strings before embedding chunks to avoid potential errors
        valid_chunk_indices = [i for i, text in enumerate(chunk_texts) if text.strip()]
        valid_chunk_texts = [chunk_texts[i] for i in valid_chunk_indices]

        if not valid_chunk_texts:
            logger.warning("RAG retrieval skipped: No non-empty chunk texts found.")
            return "", []

        chunk_embeddings = model.encode(
            valid_chunk_texts, convert_to_tensor=True, show_progress_bar=False
        )

        # Calculate cosine similarities
        # Ensure embeddings are on the same device for cosine_scores
        if prompt_embedding.device != chunk_embeddings.device:
            prompt_embedding = prompt_embedding.to(chunk_embeddings.device)
            logger.debug(f"Moved prompt embedding to device: {chunk_embeddings.device}")

        cosine_scores = util.pytorch_cos_sim(prompt_embedding, chunk_embeddings)[
            0
        ]  # Get the first row (scores against prompt)
        # Convert to numpy for easier handling if needed, or keep as tensor
        cosine_scores_np = cosine_scores.cpu().numpy()

        # Get top_k indices based on scores
        # Ensure we don't request more chunks than available
        actual_top_k = min(top_k, len(valid_chunk_texts))
        # Use argpartition for efficiency (finds k largest without full sort)
        # We want the indices of the k largest scores
        top_k_indices_relative = np.argpartition(cosine_scores_np, -actual_top_k)[
            -actual_top_k:
        ]
        # Sort these top k indices by score (descending)
        top_k_scores = cosine_scores_np[top_k_indices_relative]
        sorted_top_k_indices_relative = top_k_indices_relative[
            np.argsort(top_k_scores)[::-1]
        ]

        # Map relative indices back to original chunk indices
        top_k_original_indices = [
            valid_chunk_indices[i] for i in sorted_top_k_indices_relative
        ]

        # Retrieve the corresponding chunks and their IDs
        relevant_chunks = [chunks[i] for i in top_k_original_indices]
        relevant_chunk_ids = [
            chunk.get("chunk_id", f"unknown_{i}")
            for i, chunk in enumerate(relevant_chunks)
        ]  # Use get for safety

        # Concatenate text, adding metadata like chunk_id/page_num
        # relevant_text = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
        # Enhanced concatenation with metadata:
        relevant_parts = []
        for i, chunk in enumerate(relevant_chunks):
            chunk_id = chunk.get("chunk_id", "N/A")
            page_num = chunk.get("page_num", "N/A")  # 0-indexed
            score = cosine_scores_np[
                top_k_original_indices[i]
            ]  # Get the score for this chunk
            relevant_parts.append(
                f"[Chunk ID: {chunk_id}, Page: {page_num + 1}, Score: {score:.4f}]\n{chunk.get('text', '')}"
            )  # 1-based page for display
        relevant_text = "\n\n---\n\n".join(relevant_parts)

        logger.info(
            f"Retrieved {len(relevant_chunks)} relevant chunks for RAG. "
            f"Combined text length: {len(relevant_text)} chars."
        )
        # logger.debug(f"Top {len(relevant_chunk_ids)} Chunk IDs: {relevant_chunk_ids}") # Optional: Log retrieved IDs

        return relevant_text, relevant_chunk_ids

    except Exception as e:
        logger.error(f"Error during RAG retrieval: {e}", exc_info=True)
        # Fallback: Return empty string and list
        return "", []


# --- AI Analyzer Class ---
_thread_local = threading.local()

class DocumentAnalyzer:
    def __init__(self):
        pass  # Lazy initialization

    def _ensure_client(self):
        """Ensure that the Google client is initialized for the current thread."""
        if not hasattr(_thread_local, "google_client"):
            try:
                # Use st.secrets for the Google API Key
                if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
                    logger.info("Using Google API key from Streamlit secrets.")
                else:
                    # Fallback logic (e.g., environment variables)
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError(
                            "Google API Key is missing. Check Streamlit secrets or "
                            "GOOGLE_API_KEY environment variable."
                        )
                    logger.info("Using Google API key from environment variable.")

                genai.configure(api_key=api_key)
                # Create the model instance - adjust model name as needed
                # See https://ai.google.dev/models/gemini
                # Use newer gemini-1.5-flash model
                _thread_local.google_client = genai.GenerativeModel(
                    "gemini-1.5-flash"
                )
                logger.info(
                    f"Initialized Google GenAI client for thread {threading.current_thread().name} "
                    f"with model: {_thread_local.google_client.model_name}"
                )

            except Exception as e:
                logger.error(f"Error initializing Google AI client: {str(e)}")
                raise
        return _thread_local.google_client

    async def _get_completion(
        self, messages: List[Dict[str, str]], model: str = "gemini-1.5-flash"
    ) -> str:
        """Helper method to get completion from the Google model."""
        # NOTE: Model name might need adjustment based on availability
        try:
            client = self._ensure_client()
            # logger.debug(f"Formatted messages for API: {json.dumps(messages, indent=2)}") # Use DEBUG level

            # Gemini API prefers a direct list of content for history,
            # and system instructions can be passed separately.
            history = []
            system_instruction = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    system_instruction = content  # Store it
                elif role == "user":
                    history.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant" or role == "model": # Accept 'model' role too
                    history.append({"role": "model", "parts": [{"text": content}]})

            # ***** ADD BACK: Prepend system instruction logic *****
            if system_instruction:
                 if history and history[0]['role'] == 'user':
                      history[0]['parts'][0]['text'] = f"{system_instruction}\\n\\n---\\n\\n{history[0]['parts'][0]['text']}"
                      logger.warning("Prepending system prompt to first user message for Google API.")
                 else: # Or add as a separate user message if history starts with model or is empty
                      history.insert(0, {'role': 'user', 'parts': [{'text': system_instruction}]})
                      logger.warning("Inserting system prompt as first user message for Google API.")
            # ***** END ADD BACK *****

            logger.info(f"Sending request to Google model: {client.model_name}")
            # logger.debug(f"Formatted history for Google API: {json.dumps(history, indent=2)}")

            # Use generate_content_async without system_instruction parameter
            response = await client.generate_content_async(
                history,
                # REMOVED: system_instruction=system_instruction,
                generation_config=genai.types.GenerationConfig(
                    # candidate_count=1, # Default is 1
                    # stop_sequences=['...'],
                    max_output_tokens=8192, # Increased for Gemini 1.5 Flash
                    temperature=0.1,
                ),
            )

            # Handle potential safety blocks or empty responses
            if not response.candidates:
                safety_info = (
                    response.prompt_feedback
                    if hasattr(response, "prompt_feedback")
                    else "No specific feedback."
                )
                logger.error(
                    f"Google API returned no candidates. Possibly blocked. Feedback: {safety_info}"
                )
                raise ValueError(
                    f"Google API returned no candidates. Content may have been blocked. Feedback: {safety_info}"
                )

            content = response.text  # Access text directly
            logger.info(f"Received response from Google model {client.model_name}")
            # logger.debug(f"Raw Google API response content: {content}")
            return content

        except Exception as e:
            logger.error(
                f"Error getting completion from Google model: {str(e)}", exc_info=True
            )
            # Attempt to get more detailed error if available
            if hasattr(e, "response") and hasattr(e.response, "text"):
                logger.error(f"Google API Error Response Text: {e.response.text}")
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
                        "Exact quote 2, potentially longer...",
                        # NOTE: No chunk_id requested anymore
                    ],
                    "Context": "Optional context about this section (e.g., clause number)",
                },
                "another_descriptive_name": {
                    "Analysis": "Analysis text here...",
                    "Supporting_Phrases": [
                        "Exact quote 3...",
                        # Use "No relevant phrase found." if applicable
                    ],
                    "Context": "Optional context",
                },
                # Add more sections as identified by the AI
            },
        }

    async def analyze_document(
        self, relevant_document_text: str, filename: str, user_prompt: str
    ) -> str:  # Renamed parameter
        """Analyzes the *relevant document excerpts* based on the user prompt using RAG."""
        try:
            if not relevant_document_text:
                logger.warning(
                    f"Skipping AI analysis for {filename}: No relevant text provided "
                    "(RAG retrieval likely found nothing)."
                )
                # Return a structured message indicating no analysis was performed
                return json.dumps(
                    {
                        "title": f"Analysis Skipped for {filename}",
                        "analysis_sections": {
                            "info": {
                                "Analysis": "Analysis skipped because no relevant text sections were identified based on the prompt.",
                                "Supporting_Phrases": ["No relevant phrase found."],
                                "Context": "RAG Retrieval Found No Matches",
                            }
                        },
                    },
                    indent=2,
                )

            schema_str = json.dumps(self.output_schema_analysis, indent=2)

            # ***** MODIFIED SYSTEM PROMPT FOR RAG *****
            system_prompt = f"""You are an intelligent document analyser specializing in legal and financial documents. You will be given **relevant excerpts** from a document, identified based on the user's prompt, rather than the full text. Your task is to analyze ONLY these provided excerpts based on the user's prompt and provide structured output following a specific JSON schema.

### Core Instructions:
1.  **Analyze Thoroughly:** Read the user prompt and the document excerpts carefully. Perform the requested analysis.
2.  **Strict JSON Output:** Your entire response MUST be a single JSON object matching the schema provided below. Do not include any introductory text, explanations, apologies, or markdown formatting (` ```json`, ` ``` `) outside the JSON structure.
3.  **Descriptive Section Names:** Use lowercase snake_case for keys within `analysis_sections` (e.g., `cancellation_rights`, `liability_limitations`). These names should accurately reflect the content of the analysis in that section.
4.  **Exact Supporting Phrases:** The `Supporting_Phrases` array must contain *only direct, verbatim quotes* from the 'Relevant Document Excerpts'. Preserve original case, punctuation, and formatting within the quotes. Pay attention to the metadata (like Chunk ID/Page/Score) provided with each excerpt block, but do *not* include this metadata within the quote itself. Aim for complete sentences or meaningful clauses from the *text* part of the excerpt.
5.  **No Phrase Found:** If no relevant phrase *within the provided excerpts* directly supports the analysis point for a section, include the exact string "No relevant phrase found." in the `Supporting_Phrases` array for that section.
6.  **Focus *Only* on Excerpts:** Base your analysis *exclusively* on the text provided under '### Relevant Document Excerpts:'. Do not infer information not present in these specific excerpts.
7.  **Legal/Financial Context:** Pay attention to clause numbers, definitions, conditions, exceptions, and precise wording common in legal/financial texts found *within the excerpts*.

### JSON Output Schema:
```json
{schema_str}
```
"""

            human_prompt = f"""Please analyze the following document based on the user prompt, using ONLY the relevant excerpts provided below.

Document Name:
{filename}

User Prompt:
{user_prompt}

Relevant Document Excerpts:
--- START EXCERPTS ---
{relevant_document_text}
--- END EXCERPTS ---

Generate the analysis and supporting phrases strictly following the JSON schema provided in the system instructions. Ensure supporting phrases are exact quotes from the TEXT portion of the excerpts provided above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]

            logger.info(f"Sending RAG analysis request for {filename} to AI.")
            # Add DEBUG log for the relevant text being sent
            logger.info(f"Relevant text being sent to AI for {filename}:\n---\n{relevant_document_text}\n---")
            # logger.debug(f"AI Analysis Request Messages: {json.dumps(messages, indent=2)}") # Careful logging potentially large excerpts

            response_content = await self._get_completion(messages)
            logger.info(f"Received AI analysis response for {filename} after RAG.")

            # Attempt to clean and parse the JSON
            try:
                # Basic cleaning: remove potential markdown fences and strip whitespace
                cleaned_response = response_content.strip()
                match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
                if match:
                    cleaned_response = match.group(1)
                elif cleaned_response.startswith("```json"):
                     cleaned_response = cleaned_response[7:]
                     if cleaned_response.endswith("```"):
                         cleaned_response = cleaned_response[:-3]
                elif cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                    pass # Assume it's already JSON
                else:
                    # Fallback: try to find the first '{' and last '}'
                    first_brace = cleaned_response.find('{')
                    last_brace = cleaned_response.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        cleaned_response = cleaned_response[first_brace:last_brace+1]
                        logger.warning("Used basic brace finding for JSON extraction.")
                    else:
                         raise json.JSONDecodeError("Could not find JSON structure.", cleaned_response, 0)


                cleaned_response = cleaned_response.strip()

                # Validate if it's valid JSON
                parsed_json = json.loads(cleaned_response)

                # Optional: Validate against the schema structure (basic check)
                if "analysis_sections" not in parsed_json:
                    logger.error("AI response missing 'analysis_sections' key.")
                    raise ValueError("AI response missing 'analysis_sections' key.")
                if not isinstance(parsed_json["analysis_sections"], dict):
                    logger.error(
                        "'analysis_sections' in AI response is not a dictionary."
                    )
                    raise ValueError(
                        "'analysis_sections' in AI response is not a dictionary."
                    )

                logger.info("Successfully parsed AI analysis JSON response.")
                return json.dumps(
                    parsed_json, indent=2
                )  # Return the validated/parsed JSON string

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse AI analysis response as JSON: {json_err}")
                logger.error(f"Raw response content was:\n{response_content}")
                # Use the regex fallback only if primary parsing fails
                match = re.search(r"\{.*\}", response_content, re.DOTALL)
                if match:
                    logger.warning("Attempting to extract JSON using regex fallback.")
                    try:
                        extracted_json_str = match.group(0)
                        parsed_json = json.loads(extracted_json_str)
                        if "analysis_sections" in parsed_json:  # Basic check again
                            logger.info(
                                "Successfully parsed AI analysis JSON using regex fallback."
                            )
                            return json.dumps(parsed_json, indent=2)
                        else:
                            raise ValueError("Extracted JSON missing 'analysis_sections'.")
                    except Exception as fallback_err:
                        logger.error(
                            f"Regex JSON extraction fallback also failed: {fallback_err}"
                        )
                        raise ValueError(
                            f"AI response was not valid JSON and fallback extraction failed. "
                            f"Raw response: {response_content[:500]}..."
                        ) from json_err
                else:
                    raise ValueError(
                        f"AI response was not valid JSON and no JSON object found via regex. "
                        f"Raw response: {response_content[:500]}..."
                    ) from json_err
            except ValueError as val_err:
                # Catch schema validation errors or other ValueErrors
                logger.error(f"Error validating AI response structure: {val_err}")
                raise  # Re-raise the validation error

        except Exception as e:
            logger.error(
                f"Error during AI document analysis for {filename}: {str(e)}",
                exc_info=True,
            )
            # Return an error JSON structure
            error_response = {
                "title": f"Error Analyzing {filename}",
                "analysis_sections": {
                    "error": {
                        "Analysis": f"An error occurred during analysis: {str(e)}",
                        "Supporting_Phrases": ["No relevant phrase found."],
                        "Context": "System Error",
                    }
                },
            }
            return json.dumps(error_response, indent=2)

    # --- Keyword generation is removed as RAG uses the prompt directly ---
    # async def generate_keywords(self, prompt: str) -> List[str]: ...


# --- Document Processors ---
class PDFProcessor:
    """Handles PDF processing, chunking, verification, and annotation."""

    def __init__(self, pdf_bytes: bytes):
        if not isinstance(pdf_bytes, bytes):
            raise ValueError("pdf_bytes must be of type bytes")
        self.pdf_bytes = pdf_bytes
        self._chunks: List[Dict[str, Any]] = []
        self._full_text: Optional[str] = None
        self._processed = False # Flag to track if extraction ran
        logger.info(f"PDFProcessor initialized with {len(pdf_bytes)} bytes.")

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        if not self._processed:
            self.extract_structured_text_and_chunks()  # Lazy extraction
        return self._chunks

    # Keep full_text property in case it's needed elsewhere
    @property
    def full_text(self) -> str:
        if not self._processed:
            self.extract_structured_text_and_chunks()  # Lazy extraction
        return self._full_text if self._full_text is not None else ""

    def extract_structured_text_and_chunks(self) -> Tuple[List[Dict[str, Any]], str]:
        """Extracts text using PyMuPDF blocks and groups them into chunks."""
        if self._processed:  # Already processed
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
                    blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by top->bottom, left->right

                    current_chunk_text = []
                    current_chunk_bboxes = []
                    last_y1 = 0

                    for b in blocks:
                        x0, y0, x1, y1, text, block_no, block_type = b
                        block_text = text.strip()
                        if not block_text:  # Skip empty blocks
                            continue

                        block_rect = fitz.Rect(x0, y0, x1, y1)

                        # --- Paragraph Grouping Logic ---
                        vertical_gap = y0 - last_y1 if current_chunk_text else 0
                        # Start a new chunk if it's the first block, or a significant vertical gap exists
                        is_new_paragraph = not current_chunk_text or (vertical_gap > 8) # Adjust threshold if needed

                        if is_new_paragraph and current_chunk_text:
                            # Save the previous chunk
                            chunk_text_content = " ".join(current_chunk_text).strip()
                            if len(normalize_text(chunk_text_content)) > 50: # Min chunk length
                                chunk_id = f"chunk_{current_chunk_id}"
                                self._chunks.append(
                                    {
                                        "chunk_id": chunk_id,
                                        "text": chunk_text_content,
                                        "page_num": page_num,  # 0-indexed
                                        "bboxes": current_chunk_bboxes,
                                    }
                                )
                                all_text_parts.append(chunk_text_content)
                                current_chunk_id += 1
                            # Start new chunk
                            current_chunk_text = [block_text]
                            current_chunk_bboxes = [block_rect]
                        else:
                            # Add to the current chunk
                            current_chunk_text.append(block_text)
                            current_chunk_bboxes.append(block_rect)

                        last_y1 = y1 # Update last y-coordinate

                    # Save the last chunk of the page
                    if current_chunk_text:
                        chunk_text_content = " ".join(current_chunk_text).strip()
                        if len(normalize_text(chunk_text_content)) > 50: # Min chunk length
                            chunk_id = f"chunk_{current_chunk_id}"
                            self._chunks.append(
                                {
                                    "chunk_id": chunk_id,
                                    "text": chunk_text_content,
                                    "page_num": page_num,  # 0-indexed
                                    "bboxes": current_chunk_bboxes,
                                }
                            )
                            all_text_parts.append(chunk_text_content)
                            current_chunk_id += 1

                except Exception as page_err:
                    logger.error(f"Error processing page {page_num}: {page_err}")

            self._full_text = "\n\n".join(all_text_parts)
            self._processed = True # Mark as processed
            logger.info(
                f"Extraction complete. Generated {len(self._chunks)} chunks. "
                f"Total text length: {len(self._full_text or '')} chars."
            )

        except Exception as e:
            logger.error(f"Failed to extract text/chunks: {str(e)}", exc_info=True)
            self._full_text = ""  # Ensure empty string on failure
            self._chunks = []
            self._processed = True # Mark as processed even on failure to avoid retry loops

        finally:
            if doc:
                doc.close()
        return self._chunks, self._full_text if self._full_text is not None else ""


    def verify_and_locate_phrases(
        self, ai_analysis_json_str: str
    ) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
        """Verifies AI phrases against chunks using fuzzy matching and locates them."""
        verification_results = {}
        phrase_locations = {}

        chunks_data = self.chunks # Access property to ensure extraction runs if needed
        if not chunks_data:
            logger.warning("No chunks available for verification.")
            return {}, {}

        try:
            ai_analysis = json.loads(ai_analysis_json_str)
            if "error" in ai_analysis.get("analysis_sections", {}):
                logger.warning(
                    "AI analysis contains an error, skipping phrase verification."
                )
                return {}, {}
            if "info" in ai_analysis.get("analysis_sections", {}):
                logger.info(
                    "AI analysis skipped (RAG found no relevant text), "
                    "skipping phrase verification."
                )
                return {}, {}

            phrases_to_verify = set() # Use set to automatically handle duplicates
            # Extract all supporting phrases from the AI analysis
            for section_data in ai_analysis.get("analysis_sections", {}).values():
                if isinstance(section_data, dict):
                    phrases = section_data.get("Supporting_Phrases", [])
                    if isinstance(phrases, list):
                        for phrase in phrases:
                            p_text = ""
                            if isinstance(phrase, dict): # Should not happen based on schema, but handle defensively
                                p_text = phrase.get("text", "")
                            elif isinstance(phrase, str):
                                p_text = phrase
                            p_text = p_text.strip()
                            if p_text and p_text != "No relevant phrase found.":
                                phrases_to_verify.add(p_text)

            if not phrases_to_verify:
                logger.info("No supporting phrases found in AI analysis to verify.")
                return {}, {}

            logger.info(
                f"Starting verification for {len(phrases_to_verify)} unique phrases "
                f"against {len(chunks_data)} original chunks."
            )

            # Pre-normalize chunk texts
            normalized_chunks = [
                (chunk, normalize_text(chunk["text"])) for chunk in chunks_data if chunk.get("text")
            ]

            # --- Location Finding requires the document to be open ---
            doc = None
            try:
                doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")

                for original_phrase in phrases_to_verify:
                    verification_results[original_phrase] = False  # Initialize
                    phrase_locations[original_phrase] = []
                    normalized_phrase = normalize_text(
                        remove_markdown_formatting(original_phrase)
                    )
                    if not normalized_phrase:
                        continue

                    found_match_for_phrase = False
                    best_score_for_phrase = 0

                    # Verify against ALL original chunks
                    for chunk, norm_chunk_text in normalized_chunks:
                        if not norm_chunk_text:
                            continue

                        # --- Fuzzy Match ---
                        # Use partial_ratio or token_set_ratio
                        score = fuzz.partial_ratio(normalized_phrase, norm_chunk_text)
                        # Alternative: score = fuzz.token_set_ratio(normalized_phrase, norm_chunk_text)

                        if score >= FUZZY_MATCH_THRESHOLD:
                            if not found_match_for_phrase:  # Log first verification
                                logger.info(
                                    f"Verified (Score: {score}): '{original_phrase[:60]}...' "
                                    f"potentially in chunk {chunk['chunk_id']}"
                                )
                            found_match_for_phrase = True
                            verification_results[original_phrase] = True
                            best_score_for_phrase = max(best_score_for_phrase, score)

                            # --- Precise Location Search using PyMuPDF ---
                            page_num = chunk["page_num"]
                            if 0 <= page_num < doc.page_count:
                                page = doc[page_num]
                                # Use the union of bounding boxes as the clip area
                                clip_rect = fitz.Rect() # Empty rect
                                for bbox in chunk.get('bboxes', []):
                                    try:
                                        # Ensure bbox is a valid Rect object or tuple/list
                                        if isinstance(bbox, fitz.Rect):
                                            clip_rect.include_rect(bbox)
                                        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                            clip_rect.include_rect(fitz.Rect(bbox))
                                        # else: logger.warning(...) # Option: log invalid bbox format
                                    except Exception as bbox_err:
                                         logger.warning(f"Skipping invalid bbox {bbox} in chunk {chunk['chunk_id']}: {bbox_err}")


                                if not clip_rect.is_empty:
                                    try:
                                        # 1. Clean the phrase before searching
                                        cleaned_search_phrase = remove_markdown_formatting(original_phrase)
                                        cleaned_search_phrase = re.sub(r"\s+", " ", cleaned_search_phrase).strip()

                                        # Search for the CLEANED ORIGINAL phrase within the chunk's area
                                        instances = page.search_for(
                                            cleaned_search_phrase,
                                            clip=clip_rect,
                                            quads=False, # Usually faster than quads=True
                                        )

                                        if instances:
                                            logger.debug(
                                                f"Found {len(instances)} instance(s) via search_for in chunk "
                                                f"{chunk['chunk_id']} area for '{cleaned_search_phrase[:60]}...'"
                                            )
                                            for rect in instances:
                                                # Check if rect is valid before adding
                                                if isinstance(rect, fitz.Rect) and not rect.is_empty:
                                                    phrase_locations[original_phrase].append(
                                                        {
                                                            "page_num": page_num, # 0-indexed
                                                            "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                                                            "chunk_id": chunk["chunk_id"],
                                                            "match_score": score, # Store chunk score with exact location
                                                            "method": "exact_cleaned_search",
                                                        }
                                                    )
                                                # else: logger.warning(...) # Option: log invalid rect
                                        else:
                                            # 2. Fallback to chunk bounding box if exact search fails after fuzzy chunk match
                                            logger.debug(
                                                f"Exact search failed for '{cleaned_search_phrase[:60]}...' in chunk "
                                                f"{chunk['chunk_id']} area (score: {score}). Falling back to chunk bbox."
                                            )
                                            phrase_locations[original_phrase].append(
                                                {
                                                    "page_num": page_num,
                                                    "rect": [clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1],
                                                    "chunk_id": chunk["chunk_id"],
                                                    "match_score": score, # Use the chunk score
                                                    "method": "fuzzy_chunk_fallback",
                                                }
                                            )
                                    except Exception as search_err:
                                        logger.error(
                                            f"Error during search_for or fallback in chunk {chunk['chunk_id']}: {search_err}"
                                        )
                                # else: logger.warning(...) # Option: log empty clip_rect
                            # else: logger.warning(...) # Option: log invalid page_num

                    if not found_match_for_phrase:
                        logger.warning(
                            f"NOT Verified: '{original_phrase[:60]}...' did not meet fuzzy threshold "
                            f"({FUZZY_MATCH_THRESHOLD}) in any chunk."
                        )
            finally:
                if doc:
                    doc.close()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI analysis JSON for verification: {e}")
        except Exception as e:
            logger.error(
                f"Error during phrase verification and location: {str(e)}", exc_info=True
            )

        return verification_results, phrase_locations

    def add_annotations(
        self, phrase_locations: Dict[str, List[Dict[str, Any]]]
    ) -> bytes:
        """Adds highlights to the PDF based on found phrase locations."""
        if not phrase_locations:
            logger.warning(
                "No phrase locations provided for annotation. Returning original PDF bytes."
            )
            return self.pdf_bytes

        doc = None
        try:
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            annotated_count = 0
            highlight_color = [1, 0.9, 0.3]  # Yellow
            fallback_color = [0.5, 0.7, 1.0]  # Light Blue for fallback/chunk bbox

            for phrase, locations in phrase_locations.items():
                # Sort locations by score (desc) and method preference before annotating? Maybe not necessary.
                for loc in locations:
                    try:
                        page_num = loc.get("page_num") # Use get for safety
                        rect_coords = loc.get("rect")
                        method = loc.get("method", "unknown")

                        # Validate data before proceeding
                        if page_num is None or rect_coords is None:
                            logger.warning(
                                f"Skipping annotation due to missing page_num or rect for phrase "
                                f"'{phrase[:50]}...': {loc}"
                            )
                            continue

                        if 0 <= page_num < doc.page_count:
                            page = doc[page_num]
                            rect = fitz.Rect(rect_coords)
                            if not rect.is_empty:
                                # Use fallback color if the method indicates a less precise location
                                color = (
                                    fallback_color
                                    if method in ["fuzzy_chunk_fallback", "fuzzy_chunk_bbox"]
                                    else highlight_color
                                )
                                highlight = page.add_highlight_annot(rect)
                                highlight.set_colors(stroke=color)
                                highlight.set_info(
                                    content=(
                                        f"Verified ({method}, Score: {loc.get('match_score', 'N/A'):.0f}): "
                                        f"{phrase[:100]}..."
                                    )
                                ) # Add score info
                                highlight.update(opacity=0.4)
                                annotated_count += 1
                            # else: logger.warning(...) # Option: log skipping empty rect
                        # else: logger.warning(...) # Option: log invalid page num
                    except Exception as annot_err:
                        logger.error(
                            f"Error adding annotation for phrase '{phrase[:50]}...' at {loc}: {annot_err}"
                        )

            if annotated_count > 0:
                logger.info(f"Added {annotated_count} highlight annotations.")
                # Use memory_buffer for saving to avoid disk I/O if not needed
                annotated_bytes = doc.tobytes(garbage=4, deflate=True) # Save directly to bytes
            else:
                logger.warning(
                    "No annotations were successfully added. Returning original PDF bytes."
                )
                annotated_bytes = self.pdf_bytes

            return annotated_bytes

        except Exception as e:
            logger.error(f"Failed to add annotations: {str(e)}", exc_info=True)
            return self.pdf_bytes  # Return original on error
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
        logger.warning(
            "Using basic DOCX to PDF conversion (text dump). Formatting will be lost."
        )
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
                return self.create_minimal_empty_pdf()  # Return empty PDF

            # 2. Create basic PDF using fitz
            pdf_doc = fitz.open()
            page = pdf_doc.new_page()
            # Define text area slightly inset from page edges
            rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
            fontsize = 10  # Adjust as needed
            res = page.insert_textbox(
                rect, extracted_text, fontsize=fontsize, align=fitz.TEXT_ALIGN_LEFT
            )
            if res < 0:
                logger.warning(
                    f"Text might have been truncated during basic PDF creation (return code: {res}). "
                    "Consider a more robust DOCX->PDF converter if formatting is critical."
                )

            output_buffer = BytesIO()
            pdf_doc.save(output_buffer, garbage=4, deflate=True)  # Add compression
            pdf_doc.close()
            logger.info("Successfully created basic PDF from DOCX text.")
            return output_buffer.getvalue()

        except ImportError:
            logger.error("python-docx not installed. Cannot process Word files.")
            # Ensure python-docx is in requirements.txt
            st.error("python-docx is required to process Word documents. Please install it (`pip install python-docx`) and restart.")
            raise Exception("python-docx is required to process Word documents.") # Re-raise
        except Exception as e:
            logger.error(f"Error during basic DOCX to PDF conversion: {e}", exc_info=True)
            return self.create_minimal_empty_pdf()  # Return empty PDF on error

    def create_minimal_empty_pdf(self) -> bytes:
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
            return (
                b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n"
                b"0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n"
                b"trailer<</Size 3/Root 1 0 R>>\nstartxref\n109\n%%EOF\n"
            )

# --- Streamlit UI Functions ---

# --- PDF Viewer Logic ---
def update_pdf_view(pdf_bytes=None, page_num=None, filename=None):
    """Updates session state for the PDF viewer."""
    # Use defaults from session state if not provided
    if pdf_bytes is None:
        pdf_bytes = st.session_state.get('pdf_bytes')
    if filename is None:
        filename = st.session_state.get('current_pdf_name', 'document.pdf')
    if not isinstance(page_num, int) or page_num < 1:
        page_num = st.session_state.get('pdf_page', 1) # Default to current page if invalid input

    # Only update state if values have actually changed
    state_changed = False
    if st.session_state.get('pdf_page') != page_num:
        st.session_state.pdf_page = page_num
        state_changed = True
    # Compare bytes carefully
    if pdf_bytes is not None and pdf_bytes != st.session_state.get('pdf_bytes'):
        st.session_state.pdf_bytes = pdf_bytes
        state_changed = True
    if st.session_state.get('current_pdf_name') != filename:
        st.session_state.current_pdf_name = filename
        state_changed = True
    if not st.session_state.get('show_pdf', False):
        st.session_state.show_pdf = True
        state_changed = True

    if state_changed:
        logger.info(
            f"Updating PDF view state: page={page_num}, filename={filename}, "
            f"show={st.session_state.show_pdf}"
        )
        # No rerun needed here, let the main display loop handle it after state update

def display_pdf_viewer():
    """Renders the PDF viewer based on session state."""
    pdf_bytes = st.session_state.get("pdf_bytes")
    show_pdf = st.session_state.get("show_pdf", False)
    current_page = st.session_state.get("pdf_page", 1)
    filename = st.session_state.get("current_pdf_name", "PDF Viewer")

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
                if fitz_doc: fitz_doc.close() # Clean up fitz doc
                return # Exit early

            # --- Navigation ---
            st.caption(f"**{filename}**") # Display filename above controls
            cols = st.columns([1, 3, 1]) # Adjust ratios for button/input spacing

            with cols[0]: # Previous Button
                prev_key = f"pdf_prev_{filename}"
                if st.button(
                    "⬅️", key=prev_key, help="Previous Page", disabled=(current_page <= 1)
                ):
                    st.session_state.pdf_page -= 1
                    logger.debug(f"Button '{prev_key}' clicked, changing page to {st.session_state.pdf_page}")
                    st.rerun() # Rerun to update view

            with cols[1]: # Page Number Input
                # Use unique key incorporating filename and total_pages to reset if doc changes
                nav_key = f"nav_{filename}_{total_pages}"
                selected_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page,
                    step=1,
                    key=nav_key,
                    label_visibility="collapsed",
                    help=f"Enter page number (1-{total_pages})",
                )
                # Check if the number input value changed and is valid
                if selected_page != current_page:
                    if 1 <= selected_page <= total_pages:
                        st.session_state.pdf_page = selected_page
                        logger.debug(f"Number input '{nav_key}' changed, changing page to {st.session_state.pdf_page}")
                        st.rerun() # Rerun to update view
                    # else: # number_input enforces min/max, so no explicit warning needed

            with cols[2]: # Next Button
                next_key = f"pdf_next_{filename}"
                if st.button(
                    "➡️", key=next_key, help="Next Page", disabled=(current_page >= total_pages)
                ):
                    st.session_state.pdf_page += 1
                    logger.debug(f"Button '{next_key}' clicked, changing page to {st.session_state.pdf_page}")
                    st.rerun() # Rerun to update view

            st.caption(f"Page {current_page} of {total_pages}")

            # --- Render Page ---
            # Option 1: Render using fitz directly (avoids temp file) - Preferred
            try:
                page = fitz_doc.load_page(current_page - 1) # 0-indexed
                pix = page.get_pixmap(dpi=150) # Adjust DPI for quality/performance
                img_bytes = pix.tobytes("png") # Output as PNG bytes
                st.image(img_bytes, use_container_width=True)
            except Exception as fitz_render_err:
                 logger.error(f"Error rendering page {current_page} with Fitz: {fitz_render_err}")
                 st.error(f"Error displaying page {current_page} using Fitz.")

            # Option 2: Using pdfplumber (requires temp file - kept for reference/fallback)
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            #     tmp_file.write(pdf_bytes)
            #     tmp_file_path = tmp_file.name
            # try:
            #     pdf_doc = pdfplumber.open(tmp_file_path)
            #     if 0 <= current_page - 1 < len(pdf_doc.pages):
            #         page = pdf_doc.pages[current_page - 1]
            #         img = page.to_image(resolution=150).original # resolution=150
            #         st.image(img, use_container_width=True)
            #     else:
            #         st.error(f"Cannot display page {current_page}. Page index out of range (0 to {len(pdf_doc.pages)-1}).")
            # finally:
            #     if pdf_doc:
            #         pdf_doc.close() # Ensure pdfplumber closes the file handle
            #     if tmp_file_path and os.path.exists(tmp_file_path):
            #         try:
            #             os.unlink(tmp_file_path)
            #         except PermissionError: # Handle Windows file lock issues
            #             logger.warning(f"PermissionError deleting temp PDF file {tmp_file_path}. Will retry on exit.")
            #             atexit.register(os.unlink, tmp_file_path) # Ensure cleanup on exit
            #         except Exception as e_unlink:
            #             logger.error(f"Error deleting temp PDF file {tmp_file_path}: {e_unlink}")
            #             atexit.register(os.unlink, tmp_file_path) # Ensure cleanup on exit

        except fitz.fitz.FileNotFoundError: # Specific error type
            st.error("Failed to open the PDF data. It might be corrupted or empty.")
            st.session_state.show_pdf = False # Hide viewer on error
        except Exception as e:
            logger.error(f"Error displaying PDF viewer: {e}", exc_info=True)
            st.error(f"An error occurred while displaying the PDF: {e}")
            st.session_state.show_pdf = False # Hide viewer on critical error
        finally:
            # Ensure fitz doc is closed even if rendering fails
            if fitz_doc:
                fitz_doc.close()


# --- Analysis Display Logic ---
def find_best_location(locations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the best location (prioritizing exact matches, then highest score, then earliest page)."""
    if not locations:
        return None

    def sort_key(loc):
        method = loc.get("method", "unknown")
        score = loc.get("match_score", 0)
        page_num = loc.get("page_num", 9999)  # Default high page num if missing

        # Assign a priority score based on method (lower is better)
        method_priority = {
            "exact_cleaned_search": 0,
            "fuzzy_chunk_fallback": 1,
            # Add other methods if needed
        }.get(method, 99) # Default high priority if unknown method

        # Sort by: method priority (ascending), score (descending), page number (ascending)
        return (method_priority, -score, page_num)

    # Filter out locations without essential info first
    valid_locations = [
        loc for loc in locations if "page_num" in loc and "rect" in loc
    ]
    if not valid_locations:
        logger.warning("No valid locations found to determine the best one.")
        return None

    # Sort valid locations using the custom key
    valid_locations.sort(key=sort_key)
    best_loc = valid_locations[0]
    # logger.debug(f"Best location determined: {best_loc}") # Optional: Log best location found
    return best_loc

def display_analysis_results(analysis_results: List[Dict[str, Any]]):
    """Displays the analysis sections and citations."""
    if not analysis_results:
        logger.info("No analysis results to display.")
        return

    # Create columns for analysis and PDF viewer
    analysis_col, pdf_col = st.columns([2.5, 1.5], gap="medium")

    with analysis_col:
        st.markdown("### AI Analysis Results")

        # Check if any results actually contain analysis data (not just errors/info)
        # Corrected logic: Parse JSON and check the structure
        has_real_analysis = False
        for r in analysis_results:
            if isinstance(r, dict) and "ai_analysis" in r:
                try:
                    parsed_ai_analysis = json.loads(r["ai_analysis"])
                    if isinstance(parsed_ai_analysis, dict) and "analysis_sections" in parsed_ai_analysis:
                        sections = parsed_ai_analysis["analysis_sections"]
                        # Check if sections dict exists and doesn't contain ONLY error/info keys
                        if isinstance(sections, dict) and not (
                             "error" in sections or "info" in sections
                           ):
                             # Found at least one result with valid analysis sections
                             has_real_analysis = True
                             break # No need to check further
                except json.JSONDecodeError:
                    # Ignore results where AI analysis JSON is invalid
                    logger.warning(f"Could not parse ai_analysis for {r.get('filename', 'Unknown')} during display check.")
                    continue

        if not has_real_analysis:
            st.info(
                "Processing complete, but no analysis sections were generated "
                "(check errors above or RAG results)."
            )
            # Continue to display the PDF viewer column and tools
        else:
            for i, result in enumerate(analysis_results):
                filename = result.get("filename", "Unknown File")
                ai_analysis_json_str = result.get("ai_analysis", "{}")
                verification_results = result.get("verification_results", {})
                phrase_locations = result.get("phrase_locations", {})
                # PDF bytes are decoded only when needed for the 'Go to Page' button
                annotated_pdf_b64 = result.get("annotated_pdf") # Base64 encoded string

                st.markdown(f"--- \n#### {filename}") # Add separator and filename heading

                try:
                    ai_analysis = json.loads(ai_analysis_json_str)

                    # Handle info/error messages generated by analyze_document
                    if "error" in ai_analysis.get("analysis_sections", {}):
                        error_data = ai_analysis["analysis_sections"]["error"]
                        st.error(
                            f"Analysis Error: {error_data.get('Analysis', 'Unknown error')}"
                        )
                        continue # Skip to next file
                    if "info" in ai_analysis.get("analysis_sections", {}):
                        info_data = ai_analysis["analysis_sections"]["info"]
                        st.info(
                            f"{info_data.get('Analysis', 'No analysis performed.')} "
                            f"(Context: {info_data.get('Context', 'N/A')})"
                        )
                        continue # Skip to next file

                    analysis_sections = ai_analysis.get("analysis_sections", {})
                    if not analysis_sections:
                        st.warning("No analysis sections found in the AI response.")
                        continue

                    # Display title if present
                    if ai_analysis.get("title"):
                        st.markdown(f"##### {ai_analysis['title']}") # Use smaller heading for title

                    citation_counter = 0 # Reset for each file/result
                    for section_name, section_data in analysis_sections.items():
                        # Basic validation of section_data
                        if not isinstance(section_data, dict):
                            st.warning(
                                f"Skipping invalid section data for '{section_name}'. Expected a dictionary."
                            )
                            continue

                        display_section_name = section_name.replace("_", " ").title()

                        # Use st.container with border for visual separation
                        with st.container(border=True):
                            st.subheader(display_section_name)

                            # Analysis Text
                            if section_data.get("Analysis"):
                                # Use markdown, disable HTML for safety
                                st.markdown(
                                    section_data["Analysis"], unsafe_allow_html=False
                                )

                            # Context Text
                            if section_data.get("Context"):
                                st.caption(f"Context: {section_data['Context']}")

                            # Supporting Phrases / Citations
                            supporting_phrases = section_data.get(
                                "Supporting_Phrases", []
                            )
                            if not isinstance(supporting_phrases, list):
                                st.warning(
                                    f"Invalid format for 'Supporting_Phrases' in section '{section_name}'. Expected a list."
                                )
                                supporting_phrases = [] # Treat as empty list

                            if supporting_phrases:
                                # Expand citations by default
                                with st.expander("Supporting Citations", expanded=True):
                                    has_citations_to_show = False
                                    for phrase_text in supporting_phrases:
                                        if (
                                            not isinstance(phrase_text, str)
                                            or phrase_text == "No relevant phrase found."
                                        ):
                                            # Skip 'no phrase found' entries
                                            continue

                                        has_citations_to_show = True # Found at least one real citation
                                        citation_counter += 1
                                        is_verified = verification_results.get(
                                            phrase_text, False
                                        )
                                        locations = phrase_locations.get(phrase_text, [])
                                        best_location = find_best_location(locations)

                                        # Status Icon and details
                                        status_emoji = "✅" if is_verified else "❓"
                                        status_text = (
                                            "Verified" if is_verified else "Not Verified"
                                        )
                                        score_info = (
                                            f"Score: {best_location['match_score']:.1f}"
                                            if best_location and "match_score" in best_location
                                            else ""
                                        )
                                        method_info = (
                                            f"{best_location['method']}"
                                            if best_location and "method" in best_location
                                            else ""
                                        )
                                        page_info = (
                                            f"Pg {best_location['page_num'] + 1}"
                                            if best_location and "page_num" in best_location
                                            else ""
                                        )

                                        # Display Citation using st.markdown inside a container/div
                                        # Use columns for better layout of citation text vs button
                                        cite_col, btn_col = st.columns([0.8, 0.2], gap="small") # Adjust ratio/gap

                                        with cite_col:
                                            st.markdown(
                                                f"""
                                                <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 8px 12px; margin-bottom: 8px; background-color: #f9f9f9;">
                                                    <div style="margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                                                        <span style="font-weight: bold;">Citation {citation_counter} {status_emoji}</span>
                                                        <span style="font-size: 0.8em; color: #555;">{page_info} {score_info} <span title='{method_info}'>({status_text})</span></span>
                                                    </div>
                                                    <div style="color: #333; line-height: 1.4; font-size: 0.95em;">
                                                        <i>"{phrase_text}"</i>
                                                    </div>
                                                </div>
                                                """,
                                                unsafe_allow_html=True,
                                            )

                                        # Go to Page Button (aligned in the second column)
                                        with btn_col:
                                            if (
                                                is_verified
                                                and best_location
                                                and "page_num" in best_location
                                                and annotated_pdf_b64
                                            ):
                                                page_num_0_indexed = best_location["page_num"]
                                                page_num_1_indexed = page_num_0_indexed + 1
                                                button_key = f"goto_{filename}_{section_name}_{citation_counter}" # Unique key
                                                button_label = "Go" # Shorter label

                                                # Add some top margin to align better with the citation box
                                                st.markdown(
                                                    '<div style="margin-top: 20px;"></div>',
                                                    unsafe_allow_html=True,
                                                )
                                                if st.button(
                                                    button_label,
                                                    key=button_key,
                                                    type="secondary",
                                                    help=f"Go to Page {page_num_1_indexed} in {filename}",
                                                ):
                                                    try:
                                                        pdf_bytes = base64.b64decode(annotated_pdf_b64)
                                                        # Update view expects 1-based page num
                                                        update_pdf_view(
                                                            pdf_bytes=pdf_bytes,
                                                            page_num=page_num_1_indexed,
                                                            filename=filename,
                                                        )
                                                        # Force rerun to immediately show the update in the viewer column
                                                        st.rerun()
                                                    except Exception as decode_err:
                                                        logger.error(
                                                            f"Failed to decode/set PDF for citation button: {decode_err}",
                                                            exc_info=True,
                                                        )
                                                        st.warning("Could not load PDF for this citation.")
                                            elif is_verified:
                                                # If verified but no location or PDF data
                                                st.markdown(
                                                    '<div style="margin-top: 20px; text-align: center;">',
                                                    unsafe_allow_html=True,
                                                )
                                                st.caption("Loc N/A")
                                                st.markdown("</div>", unsafe_allow_html=True)
                                            # else: (Not verified case - no button needed)

                                    if not has_citations_to_show:
                                        st.caption(
                                            "No supporting citations provided or found for this section."
                                        )

                except json.JSONDecodeError:
                    st.error(f"Failed to decode AI analysis JSON for {filename}. Raw data:")
                    st.code(ai_analysis_json_str) # Show raw problematic data
                except Exception as display_err:
                    logger.error(
                        f"Error displaying analysis for {filename}: {display_err}",
                        exc_info=True,
                    )
                    st.error(f"Error displaying analysis results for {filename}: {display_err}")

    # --- PDF Viewer and Tools Column ---
    with pdf_col:
        st.markdown("### Analysis Tools & PDF Viewer")

        # --- Chat Interface Expander ---
        with st.expander("💬 SmartChat (Beta)", expanded=False):
            # TODO: Implement or adapt chat interface logic here if needed
            st.info("Chat feature placeholder.")
            # chat_interface() # Call your chat function if available

        # --- Export Expander ---
        with st.expander("📊 Export Results", expanded=False):
            # Filter out results that were only errors/info before exporting
            exportable_results = [
                r
                for r in analysis_results
                if isinstance(r, dict)
                and "ai_analysis" in r
                and isinstance(r.get("ai_analysis"), str)
                and '"error":' not in r["ai_analysis"]
                and '"info":' not in r["ai_analysis"]
            ]

            if exportable_results:
                col1, col2 = st.columns(2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    # Flatten results for export
                    flat_data = []
                    for res in exportable_results:
                        fname = res.get("filename", "N/A")
                        try:
                            ai_data = json.loads(res.get("ai_analysis", "{}"))
                        except json.JSONDecodeError:
                            ai_data = {} # Handle potential decode error here too
                        title = ai_data.get("title", "")
                        for sec_name, sec_data in ai_data.get(
                            "analysis_sections", {}
                        ).items():
                            # Check if sec_data is a dict before accessing keys
                            if not isinstance(sec_data, dict):
                                continue
                            analysis = sec_data.get("Analysis", "")
                            context = sec_data.get("Context", "")
                            phrases = sec_data.get("Supporting_Phrases", [])
                            if not isinstance(phrases, list):
                                phrases = [] # Ensure phrases is a list

                            # If no phrases, still include the analysis row
                            if not phrases or phrases == ["No relevant phrase found."]:
                                flat_data.append(
                                    {
                                        "Filename": fname,
                                        "AI Title": title,
                                        "Section": sec_name,
                                        "Analysis": analysis,
                                        "Context": context,
                                        "Supporting Phrase": "N/A",
                                        "Verified": "N/A",
                                        "Page": "N/A",
                                        "Match Score": "N/A",
                                        "Method": "N/A",
                                    }
                                )
                            else:
                                for phrase in phrases:
                                    if not isinstance(phrase, str):
                                        continue # Skip non-strings
                                    verified = res.get("verification_results", {}).get(
                                        phrase, False
                                    )
                                    locs = res.get("phrase_locations", {}).get(phrase, [])
                                    best_loc = find_best_location(locs)
                                    page = (
                                        best_loc["page_num"] + 1
                                        if best_loc and "page_num" in best_loc
                                        else "N/A"
                                    )
                                    score = (
                                        f"{best_loc['match_score']:.1f}"
                                        if best_loc and "match_score" in best_loc
                                        else "N/A"
                                    )
                                    method = (
                                        best_loc["method"]
                                        if best_loc and "method" in best_loc
                                        else "N/A"
                                    )
                                    flat_data.append(
                                        {
                                            "Filename": fname,
                                            "AI Title": title,
                                            "Section": sec_name,
                                            "Analysis": analysis,
                                            "Context": context,
                                            "Supporting Phrase": phrase,
                                            "Verified": verified,
                                            "Page": page,
                                            "Match Score": score,
                                            "Method": method,
                                        }
                                    )

                    if not flat_data:
                        st.info("No data available to export after filtering.")
                    else:
                        df = pd.DataFrame(flat_data)
                        # Use BytesIO buffer for Excel export
                        excel_buffer = BytesIO()
                        # Use try-except for openpyxl availability
                        try:
                            df.to_excel(excel_buffer, index=False, engine="openpyxl")
                            excel_buffer.seek(0)
                            with col1:
                                st.download_button(
                                    "📥 Export Excel",
                                    excel_buffer,
                                    f"analysis_{timestamp}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="export_excel_main",
                                )
                        except ImportError:
                            logger.error(
                                "Export to Excel failed: 'openpyxl' engine not found. Please install it."
                            )
                            with col1:
                                st.warning(
                                    "Excel export requires 'openpyxl'. Install it (`pip install openpyxl`) and restart."
                                )

                        # Export Raw JSON (of filtered results)
                        word_buffer = BytesIO()
                        export_json_str = json.dumps(exportable_results, indent=2)
                        word_buffer.write(export_json_str.encode("utf-8"))
                        word_buffer.seek(0)
                        with col2:
                            st.download_button(
                                "📄 Export Raw JSON",
                                word_buffer,
                                f"analysis_raw_{timestamp}.json",
                                "application/json",
                                key="export_json_main",
                            )

                except Exception as export_err:
                    logger.error(f"Export failed: {export_err}", exc_info=True)
                    st.error(f"Export failed: {export_err}")
            else:
                st.info("No analysis results to export.")

        # --- Report Issue Expander ---
        with st.expander("⚠️ Report Issue", expanded=False):
            # Basic issue reporting form
            st.markdown("Encountered an issue? Please describe it below.")
            issue_text = st.text_area("Issue Description", key="issue_desc")
            if st.button("Submit Issue Report", key="submit_issue"):
                if issue_text:
                    # In a real app, you'd send this somewhere (e.g., email, logging service, database)
                    logger.warning(f"ISSUE REPORTED: {issue_text}")
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("Please describe the issue before submitting.")

        # --- PDF Viewer Display ---
        # This function renders the viewer based on st.session_state
        display_pdf_viewer()


# --- Main Application Logic ---

def process_file_wrapper(args):
    """Wrapper for process_file to handle args tuple for thread pool, now with RAG."""
    # Unpack arguments (order matters!)
    (
        uploaded_file_data,
        filename,
        user_prompt,
        use_advanced_extraction, # Still used? Assume yes for PDFProcessor option
    ) = args

    # Ensure embedding model is loaded before processing
    if embedding_model is None:
        logger.error(f"Skipping processing for {filename}: Embedding model not loaded.")
        return {
            "filename": filename,
            "error": "Embedding model failed to load. Cannot process.",
            "annotated_pdf": None,
            "verification_results": {},
            "phrase_locations": {},
            "ai_analysis": json.dumps(
                {"error": "Embedding model failed to load."}
            ),
        }

    logger.info(f"Thread {threading.current_thread().name} processing: {filename} using RAG.")
    try:
        file_extension = Path(filename).suffix.lower()
        processor = None
        pdf_bytes_for_processing = None # Store bytes for PDF processing steps
        original_pdf_bytes_for_annotation = None # Store original/converted PDF for annotation step

        if file_extension == ".pdf":
            pdf_bytes_for_processing = uploaded_file_data
            original_pdf_bytes_for_annotation = uploaded_file_data
            processor = PDFProcessor(pdf_bytes_for_processing)
            # TODO: If use_advanced_extraction controls something in PDFProcessor, set it here
            # processor.set_extraction_mode(use_advanced_extraction)
            chunks, _ = processor.extract_structured_text_and_chunks()

        elif file_extension == ".docx":
            word_processor = WordProcessor(uploaded_file_data)
            pdf_bytes = word_processor.convert_to_pdf_bytes()
            if not pdf_bytes:
                raise ValueError("Failed to convert DOCX to PDF.")
            pdf_bytes_for_processing = pdf_bytes # Use the converted PDF bytes for chunking/RAG
            original_pdf_bytes_for_annotation = pdf_bytes # Annotate the converted PDF
            processor = PDFProcessor(pdf_bytes_for_processing)
            chunks, _ = processor.extract_structured_text_and_chunks()

        else:
            logger.error(f"Unsupported file type skipped in wrapper: {filename}")
            # Return an error structure compatible with the rest of the flow
            return {
                "filename": filename,
                "error": f"Unsupported file type: {file_extension}",
                "annotated_pdf": None,
                "verification_results": {},
                "phrase_locations": {},
                "ai_analysis": json.dumps({"error": f"Unsupported file type: {file_extension}"}),
             }


        # --- RAG Retrieval Step ---
        if not chunks:
            logger.warning(f"No chunks extracted for {filename}, skipping RAG and analysis.")
            # Return minimal error/info structure, include original PDF if possible
            b64_pdf = base64.b64encode(original_pdf_bytes_for_annotation).decode() if original_pdf_bytes_for_annotation else None
            return {
                "filename": filename,
                "error": "No text chunks could be extracted from the document.",
                "annotated_pdf": b64_pdf, # Return original/converted PDF
                "verification_results": {},
                "phrase_locations": {},
                "ai_analysis": json.dumps(
                    {"error": "Failed to extract text chunks."}
                ),
            }

        relevant_text, relevant_chunk_ids = retrieve_relevant_chunks(
            user_prompt, chunks, embedding_model, RAG_TOP_K
        )

        # --- AI Analysis (using relevant text) ---
        analyzer = DocumentAnalyzer() # Lazy init should work per thread
        ai_analysis_json_str = run_async(
            analyzer.analyze_document(relevant_text, filename, user_prompt)
        )

        # --- Verification & Location (uses the *original* processor instance) ---
        # Verification happens against the *original* chunks to find the source location
        verification_results, phrase_locations = processor.verify_and_locate_phrases(
            ai_analysis_json_str
        )

        # --- Annotation (on the original/converted PDF bytes using a *new* processor instance) ---
        # Important: Annotate the original PDF (or the one converted from DOCX), not the potentially modified bytes used for chunking if they differ.
        # Re-initialize PDFProcessor with the bytes meant for annotation
        annotation_processor = PDFProcessor(original_pdf_bytes_for_annotation)
        annotated_pdf_bytes = annotation_processor.add_annotations(phrase_locations)


        return {
            "filename": filename,
            # Send the annotated version of the original/converted PDF
            "annotated_pdf": base64.b64encode(annotated_pdf_bytes).decode()
            if annotated_pdf_bytes
            else None,
            "verification_results": verification_results,
            "phrase_locations": phrase_locations,
            "ai_analysis": ai_analysis_json_str, # Analysis based on RAG
            "retrieved_chunk_ids": relevant_chunk_ids, # Optional: Keep track
        }

    except Exception as e:
        logger.error(f"Error in process_file_wrapper for {filename}: {str(e)}", exc_info=True)
        # Attempt to get original bytes if available for error display
        err_pdf_bytes = uploaded_file_data if file_extension == '.pdf' else None
        # TODO: Consider how to handle DOCX errors where conversion might have failed early

        b64_err_pdf = base64.b64encode(err_pdf_bytes).decode() if err_pdf_bytes else None
        return {
            "filename": filename,
            "error": str(e),
            # Try to return original bytes if possible, otherwise None
            "annotated_pdf": b64_err_pdf,
            "verification_results": {},
            "phrase_locations": {},
            "ai_analysis": json.dumps({"error": f"Failed to process: {e}"}),
        }


def display_page():
    """Main function to display the Streamlit page with RAG."""
    # --- Initialize Session State ---
    # Keep existing ones needed
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = []
    if "show_pdf" not in st.session_state: st.session_state.show_pdf = False
    if "pdf_page" not in st.session_state: st.session_state.pdf_page = 1
    if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
    if "current_pdf_name" not in st.session_state: st.session_state.current_pdf_name = None
    if "user_prompt" not in st.session_state: st.session_state.user_prompt = ""
    # Keep advanced extraction toggle if used by PDFProcessor
    if "use_advanced_extraction" not in st.session_state: st.session_state.use_advanced_extraction = False
    # Track uploaded files to avoid reprocessing identical sets unless button is clicked
    if "last_uploaded_filenames" not in st.session_state: st.session_state.last_uploaded_filenames = set()
    if "uploaded_file_objects" not in st.session_state: st.session_state.uploaded_file_objects = []


    # --- UI Layout ---
    # Logos can be added here using get_base64_encoded_image and st.markdown
    st.markdown(
        "<h1 style='text-align: center;'>SmartDocs Document Analysis (RAG Enabled)</h1>",
        unsafe_allow_html=True,
    )

    # Check if embedding model loaded successfully
    if embedding_model is None:
        st.error(
            "Embedding model failed to load. Document processing is disabled. "
            "Please check logs and dependencies."
        )
        # Optionally add instructions or links for troubleshooting
        return # Stop further UI rendering if model is essential

    # --- File Upload ---
    uploaded_files = st.file_uploader(
        "Upload PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader_rag", # Use a distinct key
    )

    # Store uploaded file objects if the set of names changes
    current_uploaded_filenames = set(f.name for f in uploaded_files) if uploaded_files else set()
    if current_uploaded_filenames != st.session_state.get('last_uploaded_filenames', set()):
        st.session_state.uploaded_file_objects = uploaded_files
        st.session_state.last_uploaded_filenames = current_uploaded_filenames
        logger.info(f"File selection changed. New files: {st.session_state.last_uploaded_filenames}")
        # Clear previous results when files change? Optional.
        # st.session_state.analysis_results = []
        # st.session_state.show_pdf = False


    # --- Analysis Inputs ---
    with st.container(border=True):
        st.subheader("Analysis Configuration")
        # Use session state directly for prompt persistence
        st.session_state.user_prompt = st.text_area(
            "Analysis Prompt",
            placeholder="Enter specific instructions for the analysis (used for RAG retrieval)...",
            height=150,
            key="prompt_input_rag", # Distinct key
            value=st.session_state.get("user_prompt", ""), # Read from state
        )

        # Keep Advanced Extraction Toggle if PDFProcessor uses it
        st.session_state.use_advanced_extraction = st.toggle(
            "Use Advanced PDF Extraction (Layout-Aware - Experimental)", # Added note
            value=st.session_state.get("use_advanced_extraction", False),
            help="For PDFs only. May improve text extraction for complex layouts but can be slower and is experimental.",
            key="advanced_extraction_toggle_rag", # Distinct key
        )


    # --- Process Button ---
    # Disable button if embedding model isn't loaded or no files/prompt
    process_button_disabled = (
        embedding_model is None
        or not st.session_state.get('uploaded_file_objects')
        or not st.session_state.get('user_prompt', '').strip()
    )
    if st.button("Process Documents", type="primary", use_container_width=True, disabled=process_button_disabled):
        # Read necessary values from session state *at the time of the button click*
        files_to_process = st.session_state.get("uploaded_file_objects", [])
        current_user_prompt = st.session_state.get("user_prompt", "")
        current_use_advanced = st.session_state.get("use_advanced_extraction", False)

        # --- Input Validation (redundant due to button disable, but good practice) ---
        if not files_to_process:
            st.warning("Please upload one or more documents.")
        elif not current_user_prompt.strip(): # Check if prompt is empty or just whitespace
            st.error("Please enter an Analysis Prompt.")
        else:
            # --- Start Processing ---
            st.session_state.analysis_results = [] # Clear previous results
            st.session_state.show_pdf = False      # Hide PDF viewer initially
            st.session_state.pdf_bytes = None       # Clear previous PDF
            st.session_state.current_pdf_name = None

            total_files = len(files_to_process)
            overall_start_time = datetime.now()
            # Preallocate results list to maintain order
            results_list = [None] * total_files

            # Prepare arguments for parallel processing
            process_args = []
            files_read_successfully = True
            for i, uploaded_file in enumerate(files_to_process):
                try:
                    # Read file bytes ONCE here in the main thread
                    file_data = uploaded_file.getvalue()
                    # Pass necessary data to the wrapper
                    process_args.append(
                        (
                            file_data,              # File content as bytes
                            uploaded_file.name,     # Filename
                            current_user_prompt,    # User prompt for RAG/Analysis
                            current_use_advanced,   # Advanced extraction flag
                        )
                    )
                    logger.debug(f"Prepared args for file: {uploaded_file.name}")
                except Exception as read_err:
                    # Handle cases where getvalue might fail (e.g., file already closed?)
                    logger.error(f"Failed to read file {uploaded_file.name}: {read_err}", exc_info=True)
                    st.error(f"Failed to read file {uploaded_file.name}. Please re-upload.")
                    # Add error placeholder to results
                    results_list[i] = {"filename": uploaded_file.name, "error": f"Failed to read file: {read_err}"}
                    files_read_successfully = False # Indicate at least one file failed reading

            # Only proceed with processing if files were prepared successfully
            if files_read_successfully and process_args:
                files_to_run_count = len(process_args)
                # --- Use st.spinner for simpler progress indication ---
                with st.spinner(
                    f"Processing {files_to_run_count} document(s)... "
                    "(Extracting, Retrieving Relevant Chunks, Analyzing)"
                ):
                    # Define the function to run in threads (NO st calls inside)
                    def run_process_task(item_index: int, args_tuple: tuple):
                        """Processes a single file in a background thread."""
                        filename = args_tuple[1] # Get filename from tuple
                        logger.info(f"Thread {threading.current_thread().name} starting task for: {filename}")
                        try:
                            result = process_file_wrapper(args_tuple)
                            logger.info(f"Thread {threading.current_thread().name} finished task for: {filename}")
                            return item_index, result # Return original index and result
                        except Exception as thread_err:
                            logger.error(f"Unhandled error in thread task for {filename}: {thread_err}", exc_info=True)
                            # Return error structure associated with the original index
                            return item_index, {"filename": filename, "error": f"Unhandled thread error: {thread_err}"}

                    # --- Execute Tasks (Parallel or Sequential) ---
                    processed_indices = set() # Keep track of indices that have results
                    if ENABLE_PARALLEL and files_to_run_count > 1:
                        logger.info(f"Using ThreadPoolExecutor with {MAX_WORKERS} workers for {files_to_run_count} tasks.")
                        try:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                # Map future to the *original index* in the results_list
                                future_to_original_index = {
                                    executor.submit(run_process_task, i, arg_tuple): i
                                    for i, arg_tuple in enumerate(process_args)
                                }
                                # Process results as they complete
                                for future in concurrent.futures.as_completed(future_to_original_index):
                                    original_index = future_to_original_index[future]
                                    processed_indices.add(original_index)
                                    try:
                                        _, result_data = future.result() # Get (index, result) tuple
                                        results_list[original_index] = result_data # Place result in correct slot
                                    except Exception as exc:
                                        logger.error(f'Thread task execution for index {original_index} resulted in an exception: {exc}', exc_info=True)
                                        filename_for_error = process_args[original_index][1] # Get filename from original args
                                        results_list[original_index] = {"filename": filename_for_error, "error": f"Task execution failed: {exc}"}
                        except Exception as exec_err:
                            logger.error(f"ThreadPoolExecutor failed: {exec_err}", exc_info=True)
                            st.error(f"Error during parallel processing setup: {exec_err}. Try disabling parallel mode if issues persist.")
                            # Mark remaining unprocessed files as errors
                            for i in range(total_files):
                                if i not in processed_indices and results_list[i] is None:
                                    fname = files_to_process[i].name
                                    results_list[i] = {"filename": fname, "error": f"Processing cancelled due to thread pool error: {exec_err}"}

                    else: # Sequential execution
                        logger.info(f"Processing {files_to_run_count} task(s) sequentially.")
                        for i, arg_tuple in enumerate(process_args):
                            original_index = i
                            processed_indices.add(original_index)
                            try:
                                _, result_data = run_process_task(original_index, arg_tuple)
                                results_list[original_index] = result_data
                            except Exception as seq_exc:
                                 logger.error(f'Sequential task execution for index {original_index} resulted in an exception: {seq_exc}', exc_info=True)
                                 filename_for_error = arg_tuple[1]
                                 results_list[original_index] = {"filename": filename_for_error, "error": f"Task execution failed: {seq_exc}"}


                    # --- Processing Done - Update State in Main Thread ---
                    # Filter out None placeholders if any remain (shouldn't happen if logic is correct)
                    final_results = [r for r in results_list if r is not None]
                    st.session_state.analysis_results = final_results

                    total_time = (datetime.now() - overall_start_time).total_seconds()
                    # Count successes based on the final results list
                    success_count = len([r for r in final_results if isinstance(r, dict) and "error" not in r])
                    logger.info(f"Processing complete. Processed {success_count}/{total_files} files in {total_time:.2f}s.")

                    # --- Set initial PDF view (based on first successful result) ---
                    first_success_result = next((r for r in final_results if isinstance(r, dict) and "error" not in r), None)

                    if first_success_result and first_success_result.get("annotated_pdf"):
                        try:
                            pdf_bytes_decoded = base64.b64decode(first_success_result["annotated_pdf"])
                            update_pdf_view(
                                pdf_bytes=pdf_bytes_decoded,
                                page_num=1, # Start at page 1
                                filename=first_success_result.get("filename", "Unknown")
                            )
                            # State updated by update_pdf_view, rerun will handle display
                        except Exception as decode_err:
                            logger.error(f"Failed to decode/set initial PDF: {decode_err}", exc_info=True)
                            st.error("Failed to load initial PDF view.")
                            st.session_state.show_pdf = False # Ensure viewer is hidden
                    elif first_success_result:
                        logger.warning("First successful result did not contain annotated PDF data.")
                        st.warning("Processing complete, but couldn't display the first annotated document.")
                        st.session_state.show_pdf = False
                    else:
                        logger.warning("No successful results found after processing. No initial PDF view shown.")
                        st.session_state.show_pdf = False # Keep viewer hidden

            # Force a rerun AFTER processing is complete and state is updated
            st.rerun()

            # --- UI Update messages after Spinner (shown on the rerun) ---
            # These messages are now handled by the display block below the button


# --- Display Results Section (Runs on every rerun if results exist) ---
if st.session_state.get("analysis_results"):
    st.divider() # Add a visual separator
    st.markdown("## Processing Results")

    results_to_display = st.session_state.get("analysis_results", [])
    errors = [r for r in results_to_display if isinstance(r, dict) and "error" in r]
    success_results = [r for r in results_to_display if isinstance(r, dict) and "error" not in r]

    # Display overall status message based on errors/successes
    total_processed = len(results_to_display)
    if errors:
        if not success_results:
            st.error(f"Processing failed for all {total_processed} file(s). See details below.")
        else:
            st.warning(f"Processing complete for {total_processed} file(s). {len(success_results)} succeeded, {len(errors)} failed.")
    elif success_results:
        st.success(f"Successfully processed {len(success_results)} file(s).")
    # else: (No results - shouldn't happen if analysis_results is not empty)

    # Display specific errors first
    if errors:
        with st.expander("⚠️ Processing Errors", expanded=True):
            for error_res in errors:
                st.error(f"**{error_res.get('filename', 'Unknown File')}**: {error_res.get('error', 'Unknown error details.')}")

    # Display successful results using the dedicated function
    # This function creates the columns and renders the analysis/viewer/tools
    if success_results:
        display_analysis_results(success_results)
    elif not errors:
        # This case occurs if processing ran but somehow produced no results (success or error)
        st.warning("Processing finished, but no results or errors were generated.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Ensure embedding model is loaded before starting the page display
    if embedding_model is not None:
        display_page()
    else:
        # If the model failed to load initially, display_page might have already shown an error.
        # Add a fallback message here if needed, though the initial check in display_page handles it.
        logger.critical("Application cannot start because the embedding model failed to load.")
        # Check if st is available before calling it (in case of very early errors)
        if 'st' in globals():
            st.error("Critical Error: Embedding model failed to load. Application cannot start.")