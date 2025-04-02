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
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np
import pandas as pd
import pdfplumber  # For PDF viewer rendering - Can likely be removed if fitz rendering is stable
import streamlit as st
import torch  # Usually implicitly required by sentence-transformers
from docx import Document as DocxDocument  # Renamed to avoid conflict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, # Changed default level to INFO, DEBUG for detailed tracing
    format="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s" # Added threadName
)
logger = logging.getLogger(__name__)
load_dotenv()

# ****** SET PAGE CONFIG HERE (First Streamlit command) ******
st.set_page_config(layout="wide", page_title="SmartDocs Analysis")
# ****** END SET PAGE CONFIG ******

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))
ENABLE_PARALLEL = os.environ.get("ENABLE_PARALLEL", "true").lower() == "true"
FUZZY_MATCH_THRESHOLD = 88  # Adjust this threshold (0-100)
RAG_TOP_K = 10 # Number of relevant chunks to retrieve per sub-prompt (Adjusted from 15)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller, faster model for embeddings
# Consider using a faster/cheaper model for decomposition if latency is an issue
DECOMPOSITION_MODEL_NAME = "gemini-1.5-flash"
ANALYSIS_MODEL_NAME = "gemini-2.0-flash"

# --- Load Embedding Model (Cached) ---
@st.cache_resource # Use cache_resource for non-data objects like models
def load_embedding_model():
    """Loads the SentenceTransformer model and caches it."""
    model = None
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        # Check for CUDA availability, fallback to CPU if needed
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        logger.info(f"Embedding model loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        st.error(
            f"Fatal Error: Could not load embedding model '{EMBEDDING_MODEL_NAME}'. "
            "Document processing is disabled. Please check installation and dependencies."
        )
        # Return None or raise an exception if loading fails critically
        model = None
    return model

# Load the model using the cached function
embedding_model = load_embedding_model()


# --- Helper Functions ---

def normalize_text(text: Optional[str]) -> str:
    """Normalize text for comparison: lowercase, strip, whitespace."""
    if not text:
        return ""
    text = str(text)
    text = text.lower()  # Case-insensitive matching
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
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
        if loop.is_running():
            # If a loop is running, create a future and run in executor
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        logger.info("No current event loop found, creating a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
         logger.error(f"Error running async task: {e}", exc_info=True)
         raise

# Thread-safe counter class (Optional, can be removed if not used)
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
    (Modified logging slightly for clarity in sub-prompt context).
    """
    if not chunks or not prompt or model is None:
        logger.warning(f"RAG retrieval skipped for prompt '{prompt[:50]}...': No chunks, prompt, or model.")
        return "", []

    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    if not any(chunk_texts):
        logger.warning(f"RAG retrieval skipped for prompt '{prompt[:50]}...': All chunk texts are empty.")
        return "", []

    try:
        logger.info(f"RAG: Generating embeddings for {len(chunk_texts)} chunks and prompt '{prompt[:50]}...'")
        prompt_embedding = model.encode(
            prompt, convert_to_tensor=True, show_progress_bar=False
        )
        valid_chunk_indices = [i for i, text in enumerate(chunk_texts) if text.strip()]
        valid_chunk_texts = [chunk_texts[i] for i in valid_chunk_indices]

        if not valid_chunk_texts:
            logger.warning(f"RAG retrieval skipped for prompt '{prompt[:50]}...': No non-empty chunk texts.")
            return "", []

        chunk_embeddings = model.encode(
            valid_chunk_texts, convert_to_tensor=True, show_progress_bar=False
        )

        if prompt_embedding.device != chunk_embeddings.device:
            prompt_embedding = prompt_embedding.to(chunk_embeddings.device)
            logger.debug(f"Moved prompt embedding to device: {chunk_embeddings.device}")

        cosine_scores = util.pytorch_cos_sim(prompt_embedding, chunk_embeddings)[0]
        cosine_scores_np = cosine_scores.cpu().numpy()

        actual_top_k = min(top_k, len(valid_chunk_texts))
        top_k_indices_relative = np.argpartition(cosine_scores_np, -actual_top_k)[-actual_top_k:]
        top_k_scores = cosine_scores_np[top_k_indices_relative]
        sorted_top_k_indices_relative = top_k_indices_relative[np.argsort(top_k_scores)[::-1]]

        top_k_original_indices = [
            valid_chunk_indices[i] for i in sorted_top_k_indices_relative
        ]

        relevant_chunks = [chunks[i] for i in top_k_original_indices]
        relevant_chunk_ids = [
            chunk.get("chunk_id", f"unknown_{i}")
            for i, chunk in enumerate(relevant_chunks)
        ]

        relevant_parts = []
        for i, chunk_index in enumerate(top_k_original_indices):
            chunk = chunks[chunk_index]
            chunk_id = chunk.get("chunk_id", "N/A")
            page_num = chunk.get("page_num", "N/A")
            # Correctly index into cosine_scores_np using the original index mapped back
            score = cosine_scores_np[chunk_index]
            relevant_parts.append(
                f"[Chunk ID: {chunk_id}, Page: {page_num + 1}, Score: {score:.4f}]\n{chunk.get('text', '')}"
            )
        relevant_text = "\n\n---\n\n".join(relevant_parts)

        logger.info(
            f"RAG: Retrieved {len(relevant_chunks)} chunks for prompt '{prompt[:50]}...'. "
            f"Combined text length: {len(relevant_text)} chars."
        )
        # logger.debug(f"Top {len(relevant_chunk_ids)} Chunk IDs for prompt '{prompt[:50]}...': {relevant_chunk_ids}")

        return relevant_text, relevant_chunk_ids

    except Exception as e:
        logger.error(f"Error during RAG retrieval for prompt '{prompt[:50]}...': {e}", exc_info=True)
        return "", []


# --- AI Analyzer Class ---
_thread_local = threading.local()

class DocumentAnalyzer:
    def __init__(self):
        pass  # Lazy initialization

    def _ensure_client(self, model_name: str):
        """Ensure that the Google client is initialized for the current thread for the specific model."""
        # Store clients per model name in thread local storage
        if not hasattr(_thread_local, "google_clients"):
            _thread_local.google_clients = {}

        if model_name not in _thread_local.google_clients:
            try:
                if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
                    api_key = st.secrets["GOOGLE_API_KEY"]
                    logger.info(f"Using Google API key from Streamlit secrets for model {model_name}.")
                else:
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("Google API Key is missing.")
                    logger.info(f"Using Google API key from environment variable for model {model_name}.")

                genai.configure(api_key=api_key)
                # Create the specific model instance
                _thread_local.google_clients[model_name] = genai.GenerativeModel(model_name)
                logger.info(
                    f"Initialized Google GenAI client for thread {threading.current_thread().name} "
                    f"with model: {model_name}"
                )

            except Exception as e:
                logger.error(f"Error initializing Google AI client for model {model_name}: {str(e)}")
                raise
        return _thread_local.google_clients[model_name]

    async def _get_completion(
        self, messages: List[Dict[str, str]], model_name: str
    ) -> str:
        """Helper method to get completion from the specified Google model."""
        try:
            client = self._ensure_client(model_name)
            history = []
            system_instruction = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    history.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant" or role == "model":
                    history.append({"role": "model", "parts": [{"text": content}]})

            # Prepend system instruction if provided
            if system_instruction:
                 if history and history[0]['role'] == 'user':
                      history[0]['parts'][0]['text'] = f"{system_instruction}\n\n---\n\n{history[0]['parts'][0]['text']}"
                      logger.debug(f"Prepending system prompt to first user message for model {model_name}.")
                 else:
                      history.insert(0, {'role': 'user', 'parts': [{'text': system_instruction}]})
                      logger.debug(f"Inserting system prompt as first user message for model {model_name}.")

            logger.info(f"Sending request to Google model: {client.model_name}")
            # logger.debug(f"Formatted history for Google API ({client.model_name}): {json.dumps(history, indent=2)}")

            response = await client.generate_content_async(
                history,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192, # Keep high for analysis
                    temperature=0.1, # Keep low for factual tasks
                ),
                request_options={'timeout': 300} # Add timeout (in seconds)
            )

            if not response.candidates:
                safety_info = (
                    response.prompt_feedback
                    if hasattr(response, "prompt_feedback")
                    else "No specific feedback."
                )
                logger.error(
                    f"Google API ({client.model_name}) returned no candidates. Possibly blocked. Feedback: {safety_info}"
                )
                raise ValueError(
                    f"Google API ({client.model_name}) returned no candidates. Content may have been blocked. Feedback: {safety_info}"
                )

            content = response.text
            logger.info(f"Received response from Google model {client.model_name}")
            # logger.debug(f"Raw Google API ({client.model_name}) response content: {content}")
            return content

        except Exception as e:
            logger.error(
                f"Error getting completion from Google model {model_name}: {str(e)}", exc_info=True
            )
            if hasattr(e, "response") and hasattr(e.response, "text"):
                logger.error(f"Google API Error Response Text: {e.response.text}")
            # Specific check for timeout errors (common with complex tasks)
            if "Timeout" in str(e) or "DeadlineExceeded" in str(e):
                 raise TimeoutError(f"Google API request timed out for model {model_name}.") from e
            raise

    async def decompose_prompt(self, user_prompt: str) -> List[Dict[str, str]]: # Changed return type
        """
        Analyzes the user prompt with an LLM to break it down into individual questions/tasks,
        each with a suggested concise title.
        Returns a list of dictionaries, each containing 'sub_prompt' and 'title'.
        Returns [{'sub_prompt': user_prompt, 'title': 'Overall Analysis'}] on failure.
        """
        logger.info(f"Decomposing prompt: '{user_prompt[:100]}...'")
        system_prompt = """You are a helpful assistant. Your task is to analyze the user's prompt and identify distinct questions or analysis tasks within it.
Break down the prompt into a list of self-contained, individual questions or tasks. For each task, also provide a concise, descriptive title (max 5-6 words).
Your entire response MUST be a single JSON object containing a single key "decomposition", whose value is a list of JSON objects. Each object in the list must have two keys: "title" (the concise title) and "sub_prompt" (the full sub-prompt text).
Do not include any explanations, introductory text, or markdown formatting outside the JSON structure.

Example Input Prompt:
"what was the change in the median net housing value?
was there any change in The homeownership rate?
how many families had credit card debt?
how many families had student debt?"

Example JSON Output:
{
  "decomposition": [
    {
      "title": "Median Net Housing Value Change",
      "sub_prompt": "what was the change in the median net housing value?"
    },
    {
      "title": "Homeownership Rate Change",
      "sub_prompt": "was there any change in The homeownership rate?"
    },
    {
      "title": "Families with Credit Card Debt",
      "sub_prompt": "how many families had credit card debt?"
    },
    {
      "title": "Families with Student Debt",
      "sub_prompt": "how many families had student debt?"
    }
  ]
}

Example Input Prompt:
"Analyze the termination clause and liability limitations."

Example JSON Output:
{
  "decomposition": [
    {
      "title": "Termination Clause Analysis",
      "sub_prompt": "Analyze the termination clause."
    },
    {
      "title": "Liability Limitations Analysis",
      "sub_prompt": "Analyze the liability limitations."
    }
  ]
}

Example Input Prompt:
"Summarize the key findings of the report."

Example JSON Output:
{
  "decomposition": [
    {
      "title": "Key Findings Summary",
      "sub_prompt": "Summarize the key findings of the report."
    }
  ]
}
"""
        human_prompt = f"Analyze the following prompt and return the decomposed questions/tasks and their titles as a JSON list of objects according to the system instructions:\n\n{user_prompt}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        # Fallback result in case of errors
        fallback_result = [{"title": "Overall Analysis", "sub_prompt": user_prompt}]

        try:
            response_content = await self._get_completion(messages, model_name=DECOMPOSITION_MODEL_NAME)

            # Attempt to parse the JSON
            try:
                # Clean potential markdown fences (same logic as before)
                cleaned_response = response_content.strip()
                match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                elif cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                    json_str = cleaned_response
                else:
                     first_brace = cleaned_response.find('{')
                     last_brace = cleaned_response.rfind('}')
                     if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                         json_str = cleaned_response[first_brace:last_brace+1]
                         logger.warning("Used basic brace finding for JSON extraction in decomposition.")
                     else:
                         raise json.JSONDecodeError("Could not find JSON structure.", cleaned_response, 0)

                parsed_json = json.loads(json_str)

                # Validate the new structure
                if isinstance(parsed_json, dict) and "decomposition" in parsed_json:
                    decomposition_list = parsed_json["decomposition"]
                    if (isinstance(decomposition_list, list) and
                        all(isinstance(item, dict) and "title" in item and "sub_prompt" in item
                            and isinstance(item["title"], str) and isinstance(item["sub_prompt"], str)
                            for item in decomposition_list)):

                        logger.info(f"Successfully decomposed prompt into {len(decomposition_list)} sub-prompts with titles.")
                        # Filter out items with empty titles or sub-prompts
                        valid_items = [item for item in decomposition_list if item["title"].strip() and item["sub_prompt"].strip()]
                        if not valid_items:
                             logger.warning("Decomposition resulted in an empty list after filtering. Falling back.")
                             return fallback_result
                        return valid_items
                    else:
                        logger.warning("Decomposition JSON found, but 'decomposition' key is not a list of valid {'title': str, 'sub_prompt': str} objects. Falling back.")
                        return fallback_result
                else:
                    logger.warning("Decomposition JSON parsed, but missing 'decomposition' key or wrong structure. Falling back.")
                    return fallback_result

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse decomposition response as JSON: {json_err}. Raw response: {response_content}")
                logger.warning("Falling back to using the original prompt.")
                return fallback_result

        except TimeoutError:
            logger.error(f"Prompt decomposition request timed out. Falling back to original prompt.")
            return fallback_result
        except Exception as e:
            logger.error(f"Error during prompt decomposition LLM call: {str(e)}", exc_info=True)
            logger.warning("Falling back to using the original prompt.")
            return fallback_result


    @property
    def output_schema_analysis(self) -> dict:
        """Defines the expected JSON structure for document analysis."""
        # Keep this schema definition as it's used in the analysis prompt
        return {
            "title": "Concise Title for the Analysis Section based on the specific sub-prompt",
            "analysis_sections": {
                "descriptive_section_name_1": {
                    "Analysis": "Detailed analysis text for this section...",
                    "Supporting_Phrases": [
                        "Exact quote 1 from the document text...",
                        "Exact quote 2, potentially longer...",
                    ],
                    "Context": "Optional context about this section (e.g., source sub-prompt)",
                },
                # Add more sections as identified by the AI FOR THIS SUB-PROMPT
            },
        }

    async def analyze_document(
        self, relevant_document_text: str, filename: str, sub_prompt: str # Renamed parameter
    ) -> str:
        """
        Analyzes the *relevant document excerpts* based on a specific *sub-prompt* using RAG.
        Returns a JSON string containing the analysis for *this specific sub-prompt*.
        """
        try:
            if not relevant_document_text:
                logger.warning(
                    f"Skipping AI analysis for sub-prompt '{sub_prompt[:50]}...' in {filename}: No relevant text provided."
                )
                # Return a structured message indicating skipped analysis for this sub-prompt
                return json.dumps(
                    {
                        "sub_prompt_analyzed": sub_prompt,
                        "analysis_summary": f"Analysis skipped for sub-prompt '{sub_prompt}' because no relevant text sections were identified.",
                        "supporting_quotes": ["No relevant phrase found."],
                        "analysis_context": "RAG Retrieval Found No Matches"
                    },
                    indent=2,
                )

            # New simplified schema directly in the prompt
            simplified_schema = {
                "sub_prompt_analyzed": "The exact sub-prompt being analyzed",
                "analysis_summary": "Detailed analysis directly answering the sub-prompt...",
                "supporting_quotes": [
                    "Exact quote 1 from the document text...",
                    "Exact quote 2, potentially longer..."
                ],
                "analysis_context": "Optional context about the analysis (e.g., document section names)"
            }

            schema_str = json.dumps(simplified_schema, indent=2)

            # ***** MODIFIED SYSTEM PROMPT FOR RAG & SUB-PROMPT *****
            system_prompt = f"""You are an intelligent document analyser specializing in legal and financial documents. You will be given **relevant excerpts** from a document, identified based on a SPECIFIC USER SUB-PROMPT. Your task is to analyze ONLY these provided excerpts based ONLY on the given SUB-PROMPT and provide a SINGLE, FOCUSED analysis following a specific JSON schema.

### Core Instructions:
1.  **Focus on the Sub-Prompt:** Your entire analysis must address *only* the specific user SUB-PROMPT provided below. Ignore information in the excerpts not relevant to this particular sub-prompt.
2.  **Analyze Thoroughly:** Read the sub-prompt and the document excerpts carefully. Perform the requested analysis based *only* on the excerpts.
3.  **Strict JSON Output:** Your entire response MUST be a single JSON object matching the schema provided below. Do not include any introductory text, explanations, apologies, or markdown formatting (` ```json`, ` ``` `) outside the JSON structure.
4.  **Direct Answer:** Provide a *single, comprehensive analysis* that directly answers the sub-prompt. DO NOT break down your response into multiple sections or categories.
5.  **Exact Supporting Quotes:** The `supporting_quotes` array must contain *only direct, verbatim quotes* from the 'Relevant Document Excerpts'. Preserve original case, punctuation, and formatting. Do *not* include excerpt metadata (Chunk ID/Page/Score) in the quotes. Aim for complete sentences or meaningful clauses.
6.  **No Quote Found:** If no relevant phrase *within the provided excerpts* directly supports your analysis, include the exact string "No relevant phrase found." in the `supporting_quotes` array.
7.  **Focus *Only* on Excerpts:** Base your analysis *exclusively* on the text provided under '### Relevant Document Excerpts:'. Do not infer information not present in these specific excerpts.
8.  **Context Field:** Optionally use the `analysis_context` field to note the relevant section of the document.

### JSON Output Schema:
```json
{schema_str}
```
"""

            human_prompt = f"""Please analyze the following document based *only* on the user sub-prompt below, using ONLY the relevant excerpts provided.

Document Name:
{filename}

User Sub-Prompt:
{sub_prompt}

Relevant Document Excerpts (Identified for the sub-prompt above):
--- START EXCERPTS ---
{relevant_document_text}
--- END EXCERPTS ---

Generate a SINGLE, FOCUSED analysis that directly answers the sub-prompt, strictly following the JSON schema provided in the system instructions. Ensure the analysis *only* addresses this specific sub-prompt and supporting quotes are exact quotes from the TEXT portion of the excerpts provided above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ]

            logger.info(f"Sending RAG analysis request for sub-prompt '{sub_prompt[:50]}...' in {filename} to AI ({ANALYSIS_MODEL_NAME}).")
            # logger.debug(f"Relevant text being sent to AI for {filename} (sub-prompt '{sub_prompt[:50]}...'):\n---\n{relevant_document_text}\n---")

            response_content = await self._get_completion(messages, model_name=ANALYSIS_MODEL_NAME)
            logger.info(f"Received AI analysis response for sub-prompt '{sub_prompt[:50]}...' in {filename}.")

            # --- JSON Parsing and Cleaning ---
            try:
                cleaned_response = response_content.strip()
                match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned_response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                elif cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                    json_str = cleaned_response
                else:
                    first_brace = cleaned_response.find('{')
                    last_brace = cleaned_response.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str = cleaned_response[first_brace:last_brace+1]
                        logger.warning("Used basic brace finding for JSON extraction in analysis.")
                    else:
                         raise json.JSONDecodeError("Could not find JSON structure in analysis response.", cleaned_response, 0)

                parsed_json = json.loads(json_str)

                # Basic Schema Validation
                if not isinstance(parsed_json, dict):
                    raise ValueError("AI response is not a valid JSON object.")
                
                # Check for required fields in new schema
                required_fields = ["sub_prompt_analyzed", "analysis_summary", "supporting_quotes"]
                missing_fields = [field for field in required_fields if field not in parsed_json]
                
                if missing_fields:
                    logger.error(f"AI analysis response missing required fields: {missing_fields}")
                    # Attempt to salvage if possible
                    for field in missing_fields:
                        if field == "sub_prompt_analyzed":
                            parsed_json["sub_prompt_analyzed"] = sub_prompt
                        elif field == "analysis_summary":
                            parsed_json["analysis_summary"] = "Failed to generate analysis for this sub-prompt."
                        elif field == "supporting_quotes":
                            parsed_json["supporting_quotes"] = ["No relevant phrase found."]
                    
                    logger.warning(f"Salvaged analysis response by adding missing fields: {missing_fields}")
                
                # Ensure supporting_quotes is a list
                if not isinstance(parsed_json.get("supporting_quotes", []), list):
                    parsed_json["supporting_quotes"] = ["No relevant phrase found."]
                    logger.warning("Converted non-list supporting_quotes to default list.")

                logger.info(f"Successfully parsed AI analysis JSON for sub-prompt '{sub_prompt[:50]}...'.")
                return json.dumps(parsed_json, indent=2)

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse AI analysis response for sub-prompt '{sub_prompt[:50]}...' as JSON: {json_err}")
                logger.error(f"Raw response content was:\n{response_content}")
                # Return error JSON
                error_response = {
                    "sub_prompt_analyzed": sub_prompt,
                    "analysis_summary": f"Failed to parse AI response as JSON. Error: {json_err}. See logs for raw response.",
                    "supporting_quotes": ["No relevant phrase found."],
                    "analysis_context": "JSON Parsing Error"
                }
                return json.dumps(error_response, indent=2)
            except ValueError as val_err:
                logger.error(f"Error validating AI analysis response structure for sub-prompt '{sub_prompt[:50]}...': {val_err}")
                error_response = {
                    "sub_prompt_analyzed": sub_prompt,
                    "analysis_summary": f"AI response structure validation failed: {val_err}",
                    "supporting_quotes": ["No relevant phrase found."],
                    "analysis_context": "Validation Error"
                }
                return json.dumps(error_response, indent=2)

        except TimeoutError:
             logger.error(f"AI analysis request timed out for sub-prompt '{sub_prompt[:50]}...' in {filename}.")
             error_response = {
                "sub_prompt_analyzed": sub_prompt,
                "analysis_summary": f"The analysis request for this sub-prompt timed out.",
                "supporting_quotes": ["No relevant phrase found."],
                "analysis_context": "Request Timeout"
             }
             return json.dumps(error_response, indent=2)

        except Exception as e:
            logger.error(
                f"Error during AI document analysis for sub-prompt '{sub_prompt[:50]}...' in {filename}: {str(e)}",
                exc_info=True,
            )
            error_response = {
                "sub_prompt_analyzed": sub_prompt,
                "analysis_summary": f"An unexpected error occurred during analysis: {str(e)}",
                "supporting_quotes": ["No relevant phrase found."],
                "analysis_context": "System Error"
            }
            return json.dumps(error_response, indent=2)


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

            min_chunk_char_length = 50 # Chunks must be at least this long

            for page_num, page in enumerate(doc):
                try:
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: (b[1], b[0]))

                    current_chunk_text_parts = []
                    current_chunk_bboxes = []
                    last_y1 = 0

                    for b in blocks:
                        x0, y0, x1, y1, text, block_no, block_type = b
                        block_text = text.strip()
                        if not block_text:
                            continue

                        block_rect = fitz.Rect(x0, y0, x1, y1)
                        vertical_gap = y0 - last_y1 if current_chunk_text_parts else 0
                        # Simple paragraph logic: new chunk if big vertical gap
                        is_new_paragraph = not current_chunk_text_parts or (vertical_gap > 8) # Threshold

                        if is_new_paragraph and current_chunk_text_parts:
                            # Finalize previous chunk
                            chunk_text_content = " ".join(current_chunk_text_parts).strip()
                            if len(normalize_text(chunk_text_content)) >= min_chunk_char_length:
                                chunk_id = f"chunk_{current_chunk_id}"
                                self._chunks.append({
                                    "chunk_id": chunk_id,
                                    "text": chunk_text_content,
                                    "page_num": page_num,
                                    "bboxes": current_chunk_bboxes,
                                })
                                all_text_parts.append(chunk_text_content)
                                current_chunk_id += 1
                            # Start new chunk
                            current_chunk_text_parts = [block_text]
                            current_chunk_bboxes = [block_rect]
                        else:
                            # Add to current chunk
                            current_chunk_text_parts.append(block_text)
                            current_chunk_bboxes.append(block_rect)

                        last_y1 = y1

                    # Save the last chunk of the page if it meets length requirement
                    if current_chunk_text_parts:
                        chunk_text_content = " ".join(current_chunk_text_parts).strip()
                        if len(normalize_text(chunk_text_content)) >= min_chunk_char_length:
                            chunk_id = f"chunk_{current_chunk_id}"
                            self._chunks.append({
                                "chunk_id": chunk_id,
                                "text": chunk_text_content,
                                "page_num": page_num,
                                "bboxes": current_chunk_bboxes,
                            })
                            all_text_parts.append(chunk_text_content)
                            current_chunk_id += 1

                except Exception as page_err:
                    logger.error(f"Error processing page {page_num}: {page_err}")

            self._full_text = "\n\n".join(all_text_parts)
            self._processed = True
            logger.info(
                f"Extraction complete. Generated {len(self._chunks)} chunks. "
                f"Total text length: {len(self._full_text or '')} chars."
            )

        except Exception as e:
            logger.error(f"Failed to extract text/chunks: {str(e)}", exc_info=True)
            self._full_text = ""
            self._chunks = []
            self._processed = True # Mark as processed even on failure

        finally:
            if doc:
                doc.close()
        return self._chunks, self._full_text if self._full_text is not None else ""


    def verify_and_locate_phrases(
        self, ai_analysis_json_str: str # Expects the *aggregated* JSON string
    ) -> Tuple[Dict[str, bool], Dict[str, List[Dict[str, Any]]]]:
        """Verifies AI phrases from the aggregated analysis against chunks and locates them."""
        verification_results = {}
        phrase_locations = {}

        chunks_data = self.chunks
        if not chunks_data:
            logger.warning("No chunks available for verification.")
            return {}, {}

        try:
            # Parse the *aggregated* AI analysis
            ai_analysis = json.loads(ai_analysis_json_str)

            # Check if the entire analysis was just an error placeholder
            if not ai_analysis.get("analysis_sections") or \
               all(k.startswith("error_") for k in ai_analysis.get("analysis_sections", {})):
                logger.warning("AI analysis contains only errors or is empty, skipping phrase verification.")
                return {}, {}

            phrases_to_verify = set()
            # Extract all supporting phrases from *all* sections in the aggregated analysis
            for section_key, section_data in ai_analysis.get("analysis_sections", {}).items():
                 # Skip sections indicating skipped RAG or errors generated during analysis
                 if section_key.startswith("info_skipped_") or section_key.startswith("error_"):
                     continue
                 if isinstance(section_data, dict):
                    # Check for both old and new field names for supporting phrases
                    phrases = section_data.get("Supporting_Phrases", section_data.get("supporting_quotes", []))
                    if isinstance(phrases, list):
                        for phrase in phrases:
                            p_text = ""
                            if isinstance(phrase, str):
                                p_text = phrase
                            p_text = p_text.strip()
                            # Exclude the "No relevant phrase found." placeholder
                            if p_text and p_text != "No relevant phrase found.":
                                phrases_to_verify.add(p_text)

            if not phrases_to_verify:
                logger.info("No supporting phrases found in aggregated AI analysis to verify.")
                return {}, {}

            logger.info(
                f"Starting verification for {len(phrases_to_verify)} unique phrases "
                f"(from aggregated analysis) against {len(chunks_data)} original chunks."
            )

            normalized_chunks = [
                (chunk, normalize_text(chunk["text"])) for chunk in chunks_data if chunk.get("text")
            ]

            doc = None
            try:
                doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")

                for original_phrase in phrases_to_verify:
                    verification_results[original_phrase] = False
                    phrase_locations[original_phrase] = []
                    normalized_phrase = normalize_text(remove_markdown_formatting(original_phrase))
                    if not normalized_phrase: continue

                    found_match_for_phrase = False
                    best_score_for_phrase = 0

                    # Verify against ALL original chunks
                    for chunk, norm_chunk_text in normalized_chunks:
                        if not norm_chunk_text: continue

                        score = fuzz.partial_ratio(normalized_phrase, norm_chunk_text)
                        # score = fuzz.token_set_ratio(normalized_phrase, norm_chunk_text) # Alternative

                        if score >= FUZZY_MATCH_THRESHOLD:
                            if not found_match_for_phrase:
                                logger.info(f"Verified (Score: {score}) '{original_phrase[:60]}...' potentially in chunk {chunk['chunk_id']}")
                            found_match_for_phrase = True
                            verification_results[original_phrase] = True
                            best_score_for_phrase = max(best_score_for_phrase, score)

                            # --- Precise Location Search ---
                            page_num = chunk["page_num"]
                            if 0 <= page_num < doc.page_count:
                                page = doc[page_num]
                                clip_rect = fitz.Rect()
                                for bbox in chunk.get('bboxes', []):
                                    try:
                                        if isinstance(bbox, fitz.Rect): clip_rect.include_rect(bbox)
                                        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4: clip_rect.include_rect(fitz.Rect(bbox))
                                    except Exception as bbox_err: logger.warning(f"Skipping invalid bbox {bbox} in chunk {chunk['chunk_id']}: {bbox_err}")

                                if not clip_rect.is_empty:
                                    try:
                                        cleaned_search_phrase = remove_markdown_formatting(original_phrase)
                                        cleaned_search_phrase = re.sub(r"\s+", " ", cleaned_search_phrase).strip()
                                        instances = page.search_for(cleaned_search_phrase, clip=clip_rect, quads=False)

                                        if instances:
                                            logger.debug(f"Found {len(instances)} instance(s) via search_for in chunk {chunk['chunk_id']} area for '{cleaned_search_phrase[:60]}...'")
                                            for rect in instances:
                                                if isinstance(rect, fitz.Rect) and not rect.is_empty:
                                                    phrase_locations[original_phrase].append({
                                                        "page_num": page_num,
                                                        "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                                                        "chunk_id": chunk["chunk_id"],
                                                        "match_score": score,
                                                        "method": "exact_cleaned_search",
                                                    })
                                        else:
                                            # Fallback to chunk bounding box if exact search fails within the verified chunk
                                            logger.debug(f"Exact search failed for '{cleaned_search_phrase[:60]}...' in verified chunk {chunk['chunk_id']} (score: {score}). Falling back to chunk bbox.")
                                            phrase_locations[original_phrase].append({
                                                "page_num": page_num,
                                                "rect": [clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1],
                                                "chunk_id": chunk["chunk_id"],
                                                "match_score": score,
                                                "method": "fuzzy_chunk_fallback",
                                            })
                                    except Exception as search_err: logger.error(f"Error during search_for/fallback in chunk {chunk['chunk_id']}: {search_err}")
                            # else: logger.warning(f"Invalid page number {page_num} for chunk {chunk['chunk_id']}")

                    if not found_match_for_phrase:
                        logger.warning(f"NOT Verified: '{original_phrase[:60]}...' did not meet fuzzy threshold ({FUZZY_MATCH_THRESHOLD}) in any chunk.")
            finally:
                if doc: doc.close()

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse aggregated AI analysis JSON for verification: {e}")
            logger.debug(f"Problematic JSON string: {ai_analysis_json_str[:500]}...") # Log start of bad JSON
        except Exception as e:
            logger.error(f"Error during phrase verification and location: {str(e)}", exc_info=True)

        return verification_results, phrase_locations

    def add_annotations(
        self, phrase_locations: Dict[str, List[Dict[str, Any]]]
    ) -> bytes:
        """Adds highlights to the PDF based on found phrase locations (from aggregated results)."""
        if not phrase_locations:
            logger.warning("No phrase locations provided for annotation. Returning original PDF bytes.")
            return self.pdf_bytes

        doc = None
        try:
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            annotated_count = 0
            highlight_color = [1, 0.9, 0.3]  # Yellow
            fallback_color = [0.5, 0.7, 1.0]  # Light Blue for fallback

            # Flatten all locations from the dict for easier processing
            all_locs = []
            for phrase, locations in phrase_locations.items():
                for loc in locations:
                    # Add the phrase back into the location dict for context in annotation info
                    loc['phrase_text'] = phrase
                    all_locs.append(loc)

            # Optional: Sort annotations to potentially process page by page
            # all_locs.sort(key=lambda x: (x.get('page_num', -1), x.get('rect', [0,0,0,0])[1]))

            for loc in all_locs:
                try:
                    page_num = loc.get("page_num")
                    rect_coords = loc.get("rect")
                    method = loc.get("method", "unknown")
                    phrase = loc.get("phrase_text", "Unknown Phrase")

                    if page_num is None or rect_coords is None:
                        logger.warning(f"Skipping annotation due to missing page_num/rect for phrase '{phrase[:50]}...': {loc}")
                        continue

                    if 0 <= page_num < doc.page_count:
                        page = doc[page_num]
                        rect = fitz.Rect(rect_coords)
                        if not rect.is_empty:
                            color = fallback_color if "fallback" in method else highlight_color
                            highlight = page.add_highlight_annot(rect)
                            highlight.set_colors(stroke=color)
                            highlight.set_info(
                                content=(f"Verified ({method}, Score: {loc.get('match_score', 'N/A'):.0f}): {phrase[:100]}...")
                            )
                            highlight.update(opacity=0.4)
                            annotated_count += 1
                        # else: logger.debug(f"Skipping annotation for empty rect: {rect}")
                    # else: logger.warning(f"Skipping annotation due to invalid page num {page_num} from location data.")
                except Exception as annot_err:
                    logger.error(f"Error adding annotation for phrase '{phrase[:50]}...' at {loc}: {annot_err}")

            if annotated_count > 0:
                logger.info(f"Added {annotated_count} highlight annotations.")
                annotated_bytes = doc.tobytes(garbage=4, deflate=True)
            else:
                logger.warning("No annotations were successfully added. Returning original PDF bytes.")
                annotated_bytes = self.pdf_bytes

            return annotated_bytes

        except Exception as e:
            logger.error(f"Failed to add annotations: {str(e)}", exc_info=True)
            return self.pdf_bytes # Return original on error
        finally:
            if doc: doc.close()

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
            doc = DocxDocument(BytesIO(self.docx_bytes))
            full_text = []
            for para in doc.paragraphs: full_text.append(para.text)
            # Basic table extraction (join cells with tabs, rows with newlines)
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    table_text.append("\t".join(row_text))
                full_text.append("\n".join(table_text))

            extracted_text = "\n".join(full_text)

            if not extracted_text.strip():
                logger.warning("No text extracted from DOCX file.")
                return self.create_minimal_empty_pdf()

            pdf_doc = fitz.open()
            page = pdf_doc.new_page()
            rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
            fontsize = 10
            # Insert text with basic formatting interpretation
            res = page.insert_textbox(
                rect,
                extracted_text,
                fontsize=fontsize,
                fontname="helv", # Use a standard font
                align=fitz.TEXT_ALIGN_LEFT
            )
            if res < 0: logger.warning(f"Text might have been truncated during basic PDF creation (return code: {res}).")

            output_buffer = BytesIO()
            pdf_doc.save(output_buffer, garbage=4, deflate=True)
            pdf_doc.close()
            logger.info("Successfully created basic PDF from DOCX text.")
            return output_buffer.getvalue()

        except ImportError:
            logger.error("python-docx not installed. Cannot process Word files.")
            st.error("python-docx is required to process Word documents. Please install it (`pip install python-docx`) and restart.")
            raise Exception("python-docx is required to process Word documents.")
        except Exception as e:
            logger.error(f"Error during basic DOCX to PDF conversion: {e}", exc_info=True)
            return self.create_minimal_empty_pdf()

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
            # Absolute last resort (often unnecessary if fitz works)
            return (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                    b"2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n"
                    b"0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n"
                    b"trailer<</Size 3/Root 1 0 R>>\nstartxref\n109\n%%EOF\n")


# --- Streamlit UI Functions ---

# --- PDF Viewer Logic ---
def update_pdf_view(pdf_bytes=None, page_num=None, filename=None):
    """Updates session state for the PDF viewer."""
    if pdf_bytes is None: pdf_bytes = st.session_state.get('pdf_bytes')
    if filename is None: filename = st.session_state.get('current_pdf_name', 'document.pdf')
    if not isinstance(page_num, int) or page_num < 1: page_num = st.session_state.get('pdf_page', 1)

    state_changed = False
    if st.session_state.get('pdf_page') != page_num:
        st.session_state.pdf_page = page_num
        state_changed = True
    # Careful bytes comparison: only update if actually different and not None
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
        logger.info(f"Updating PDF view state: page={page_num}, filename={filename}, show={st.session_state.show_pdf}")
        # No rerun needed here, will be handled by button click or main loop

def display_pdf_viewer():
    """Renders the PDF viewer based on session state."""
    pdf_bytes = st.session_state.get("pdf_bytes")
    show_pdf = st.session_state.get("show_pdf", False)
    current_page = st.session_state.get("pdf_page", 1)
    filename = st.session_state.get("current_pdf_name", "PDF Viewer")

    # Use expander's default state based on show_pdf
    with st.expander("📄 PDF Viewer", expanded=show_pdf):
        if not pdf_bytes:
            st.info("Upload and process a file, or click a citation to view the PDF here.")
            return

        fitz_doc = None
        try:
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = fitz_doc.page_count
            if total_pages == 0:
                st.warning("The PDF document appears to have 0 pages.")
                return

            # --- Navigation ---
            st.caption(f"**{filename}**")
            cols = st.columns([1, 3, 1])

            with cols[0]:
                prev_key = f"pdf_prev_{filename}_{total_pages}" # More unique key
                if st.button("⬅️", key=prev_key, help="Previous Page", disabled=(current_page <= 1)):
                    st.session_state.pdf_page = max(1, current_page - 1)
                    st.rerun()

            with cols[1]:
                nav_key = f"nav_{filename}_{total_pages}"
                selected_page = st.number_input(
                    "Page", min_value=1, max_value=total_pages, value=current_page, step=1,
                    key=nav_key, label_visibility="collapsed", help=f"Enter page number (1-{total_pages})"
                )
                if selected_page != current_page:
                    if 1 <= selected_page <= total_pages:
                        st.session_state.pdf_page = selected_page
                        st.rerun()

            with cols[2]:
                next_key = f"pdf_next_{filename}_{total_pages}" # More unique key
                if st.button("➡️", key=next_key, help="Next Page", disabled=(current_page >= total_pages)):
                    st.session_state.pdf_page = min(total_pages, current_page + 1)
                    st.rerun()

            st.caption(f"Page {current_page} of {total_pages}")

            # --- Render Page using Fitz ---
            try:
                page = fitz_doc.load_page(current_page - 1) # 0-indexed
                pix = page.get_pixmap(dpi=150) # Adjust DPI
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, use_container_width=True)
            except Exception as fitz_render_err:
                 logger.error(f"Error rendering page {current_page} with Fitz: {fitz_render_err}")
                 st.error(f"Error displaying page {current_page}.")

        except fitz.fitz.FileNotFoundError: # More specific type
            st.error("Failed to open the PDF data. It might be corrupted or empty.")
            st.session_state.show_pdf = False # Hide viewer on error
        except Exception as e:
            logger.error(f"Error displaying PDF viewer: {e}", exc_info=True)
            st.error(f"An error occurred while displaying the PDF: {e}")
            st.session_state.show_pdf = False # Hide viewer on critical error
        finally:
            if fitz_doc: fitz_doc.close()


# --- Analysis Display Logic ---
def find_best_location(locations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Finds the best location (prioritizing exact matches, then highest score, then earliest page)."""
    if not locations: return None

    def sort_key(loc):
        method = loc.get("method", "unknown")
        score = loc.get("match_score", 0)
        page_num = loc.get("page_num", 9999)
        method_priority = {"exact_cleaned_search": 0, "fuzzy_chunk_fallback": 1}.get(method, 99)
        return (method_priority, -score, page_num)

    valid_locations = [loc for loc in locations if "page_num" in loc and "rect" in loc]
    if not valid_locations: return None
    valid_locations.sort(key=sort_key)
    return valid_locations[0]

def display_analysis_results(analysis_results: List[Dict[str, Any]]):
    """Displays the aggregated analysis sections and citations."""
    if not analysis_results:
        logger.info("No analysis results to display.")
        return

    analysis_col, pdf_col = st.columns([2.5, 1.5], gap="small")

    with analysis_col:
        st.markdown("### AI Analysis Results")

        # Check if any results contain actual analysis data (not just errors/info/skipped)
        has_real_analysis = False
        aggregated_analysis_data = {} # Store the first valid parsed analysis
        first_result_with_data = None

        for r in analysis_results:
            if isinstance(r, dict) and "ai_analysis" in r and isinstance(r['ai_analysis'], str):
                try:
                    parsed_ai_analysis = json.loads(r["ai_analysis"])
                    if isinstance(parsed_ai_analysis, dict) and "analysis_sections" in parsed_ai_analysis:
                        sections = parsed_ai_analysis["analysis_sections"]
                        # Check if sections dict exists and contains keys NOT starting with error/info/skipped
                        if isinstance(sections, dict) and any(not k.startswith(("error_", "info_", "skipped_")) for k in sections):
                             has_real_analysis = True
                             aggregated_analysis_data = parsed_ai_analysis # Store the parsed data
                             first_result_with_data = r # Store the whole result dict for PDF access
                             break # Found one, assume it's the main aggregated one
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse ai_analysis for {r.get('filename', 'Unknown')} during display check.")
                    continue

        if not has_real_analysis:
            st.info(
                "Processing complete, but no analysis sections were successfully generated "
                "(check errors or RAG results if files were processed)."
            )
            # Display specific errors from results if they exist
            for r in analysis_results:
                 if isinstance(r, dict) and "ai_analysis" in r and isinstance(r['ai_analysis'], str):
                      try:
                           parsed = json.loads(r['ai_analysis'])
                           if isinstance(parsed, dict) and parsed.get('analysis_sections'):
                                for key, section in parsed['analysis_sections'].items():
                                     if key.startswith(("error_", "info_", "skipped_")) and isinstance(section, dict):
                                          st.warning(f"**{r.get('filename', 'File')} - Info/Error:** {section.get('Analysis', 'Details unavailable.')} (Context: {section.get('Context', 'N/A')})")
                      except: pass # Ignore parsing errors here, already warned elsewhere

        else:
            # Display results from the successfully parsed aggregated data
            result = first_result_with_data # Use the result dict containing the valid analysis
            filename = result.get("filename", "Unknown File")
            # ai_analysis_json_str = result.get("ai_analysis", "{}") # Already parsed above
            verification_results = result.get("verification_results", {})
            phrase_locations = result.get("phrase_locations", {})
            annotated_pdf_b64 = result.get("annotated_pdf")

            st.markdown(f"--- \n#### {filename}")

            try:
                # Use the already parsed aggregated_analysis_data
                ai_analysis = aggregated_analysis_data

                analysis_sections = ai_analysis.get("analysis_sections", {})
                if not analysis_sections:
                    st.warning("No analysis sections found in the aggregated AI response.")
                    # Continue to PDF column

                # Display title if present in aggregated data
                if ai_analysis.get("title"):
                    st.markdown(f"##### {ai_analysis['title']}")

                citation_counter = 0
                for section_key, section_data in analysis_sections.items():
                    if not isinstance(section_data, dict): continue

                    # Skip placeholder/error sections in main display loop
                    if section_key.startswith(("error_", "info_", "skipped_")):
                         # Optionally display these differently or in the error expander later
                         st.caption(f"Skipped/Error Section: '{section_key}' - Check logs or error summary for details.")
                         continue

                    display_section_name = section_key.replace("_", " ").title()

                    with st.container(border=True):
                        st.markdown(f"##### {display_section_name}")
                        if section_data.get("Analysis"):
                            st.markdown(section_data["Analysis"], unsafe_allow_html=False)
                        if section_data.get("Context"):
                            st.caption(f"Context: {section_data['Context']}")

                        supporting_phrases = section_data.get("Supporting_Phrases", section_data.get("supporting_quotes", []))
                        if not isinstance(supporting_phrases, list): supporting_phrases = []

                        if supporting_phrases:
                            with st.expander("Supporting Citations", expanded=False):
                                has_citations_to_show = False
                                for phrase_text in supporting_phrases:
                                    if not isinstance(phrase_text, str) or phrase_text == "No relevant phrase found.":
                                        continue
                                    has_citations_to_show = True
                                    citation_counter += 1
                                    is_verified = verification_results.get(phrase_text, False)
                                    locations = phrase_locations.get(phrase_text, [])
                                    best_location = find_best_location(locations)

                                    status_emoji = "✅" if is_verified else "❓"
                                    status_text = "Verified" if is_verified else "Not Verified"
                                    score_info = f"Score: {best_location['match_score']:.1f}" if best_location and "match_score" in best_location else ""
                                    method_info = f"{best_location['method']}" if best_location and "method" in best_location else ""
                                    page_info = f"Pg {best_location['page_num'] + 1}" if best_location and "page_num" in best_location else ""

                                    cite_col, btn_col = st.columns([0.90, 0.10], gap="small")
                                    with cite_col:
                                        st.markdown(f"""
                                            <div style="border: 1px solid #e0e0e0; border-radius: 5px; padding: 8px 12px; margin-bottom: 8px; background-color: #f9f9f9;">
                                                <div style="margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                                                    <span style="font-weight: bold;">Citation {citation_counter} {status_emoji}</span>
                                                    <span style="font-size: 0.8em; color: #555;">{page_info} {score_info} <span title='{method_info}'>({status_text})</span></span>
                                                </div>
                                                <div style="color: #333; line-height: 1.4; font-size: 0.95em;"><i>"{phrase_text}"</i></div>
                                            </div>""", unsafe_allow_html=True)
                                    with btn_col:
                                        if is_verified and best_location and "page_num" in best_location and annotated_pdf_b64:
                                            page_num_1_indexed = best_location["page_num"] + 1
                                            button_key = f"goto_{filename}_{section_key}_{citation_counter}" # Unique key
                                            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                                            if st.button("Go", key=button_key, type="secondary", help=f"Go to Page {page_num_1_indexed} in {filename}"):
                                                try:
                                                    pdf_bytes = base64.b64decode(annotated_pdf_b64)
                                                    update_pdf_view(pdf_bytes=pdf_bytes, page_num=page_num_1_indexed, filename=filename)
                                                    # st.rerun() # <-- REMOVE THIS LINE
                                                except Exception as decode_err:
                                                    logger.error(f"Failed to decode/set PDF for citation button: {decode_err}", exc_info=True)
                                                    st.warning("Could not load PDF for this citation.")
                                        elif is_verified:
                                            st.markdown('<div style="margin-top: 20px; text-align: center;">', unsafe_allow_html=True)
                                            st.caption("Loc N/A")
                                            st.markdown("</div>", unsafe_allow_html=True)
                                if not has_citations_to_show:
                                    st.caption("No supporting citations provided or found for this section.")

            except Exception as display_err:
                logger.error(f"Error displaying analysis for {filename}: {display_err}", exc_info=True)
                st.error(f"Error displaying analysis results for {filename}: {display_err}")

    # --- PDF Viewer and Tools Column ---
    with pdf_col:
        st.markdown("### Analysis Tools & PDF Viewer")

        # --- Chat Interface Expander ---
        with st.expander("💬 SmartChat (Beta)", expanded=False):
            st.info("Chat feature placeholder.")

        # --- Export Expander ---
        with st.expander("📊 Export Results", expanded=False):
            # We already filtered for display, use the same logic/data
            if has_real_analysis and first_result_with_data:
                exportable_result = first_result_with_data # Use the single aggregated result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    flat_data = []
                    fname = exportable_result.get("filename", "N/A")
                    ai_data = aggregated_analysis_data # Use the parsed data
                    title = ai_data.get("title", "")
                    verif_res = exportable_result.get("verification_results", {})
                    phrase_locs = exportable_result.get("phrase_locations", {})

                    for sec_name, sec_data in ai_data.get("analysis_sections", {}).items():
                        if not isinstance(sec_data, dict) or sec_name.startswith(("error_", "info_", "skipped_")):
                            continue # Skip non-dict or placeholder sections for export
                        analysis = sec_data.get("Analysis", "")
                        context = sec_data.get("Context", "")
                        phrases = sec_data.get("Supporting_Phrases", sec_data.get("supporting_quotes", []))
                        if not isinstance(phrases, list): phrases = []

                        if not phrases or phrases == ["No relevant phrase found."]:
                            flat_data.append({"Filename": fname, "AI Title": title, "Section": sec_name, "Analysis": analysis, "Context": context, "Supporting Phrase": "N/A", "Verified": "N/A", "Page": "N/A", "Match Score": "N/A", "Method": "N/A"})
                        else:
                            for phrase in phrases:
                                if not isinstance(phrase, str): continue
                                verified = verif_res.get(phrase, False)
                                locs = phrase_locs.get(phrase, [])
                                best_loc = find_best_location(locs)
                                page = best_loc["page_num"] + 1 if best_loc and "page_num" in best_loc else "N/A"
                                score = f"{best_loc['match_score']:.1f}" if best_loc and "match_score" in best_loc else "N/A"
                                method = best_loc["method"] if best_loc and "method" in best_loc else "N/A"
                                flat_data.append({"Filename": fname, "AI Title": title, "Section": sec_name, "Analysis": analysis, "Context": context, "Supporting Phrase": phrase, "Verified": verified, "Page": page, "Match Score": score, "Method": method})

                    if not flat_data:
                        st.info("No data available to export after filtering analysis sections.")
                    else:
                        col1, col2 = st.columns(2)
                        df = pd.DataFrame(flat_data)
                        excel_buffer = BytesIO()
                        try:
                            df.to_excel(excel_buffer, index=False, engine="openpyxl")
                            excel_buffer.seek(0)
                            with col1: st.download_button("📥 Export Excel", excel_buffer, f"analysis_{fname}_{timestamp}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="export_excel_main")
                        except ImportError:
                             logger.error("Export to Excel failed: 'openpyxl' not found.")
                             with col1: st.warning("Excel export requires 'openpyxl'. Install it (`pip install openpyxl`)")

                        # Export Aggregated Raw JSON
                        json_buffer = BytesIO()
                        # Use the already parsed and validated aggregated_analysis_data
                        export_json_str = json.dumps(aggregated_analysis_data, indent=2)
                        json_buffer.write(export_json_str.encode("utf-8"))
                        json_buffer.seek(0)
                        with col2: st.download_button("📄 Export JSON", json_buffer, f"analysis_aggregated_{fname}_{timestamp}.json", "application/json", key="export_json_main")

                except Exception as export_err:
                    logger.error(f"Export failed: {export_err}", exc_info=True)
                    st.error(f"Export failed: {export_err}")
            else:
                st.info("No analysis results available to export.")

        # --- Report Issue Expander ---
        with st.expander("⚠️ Report Issue", expanded=False):
            st.markdown("Encountered an issue? Please describe it below.")
            issue_text = st.text_area("Issue Description", key="issue_desc")
            if st.button("Submit Issue Report", key="submit_issue"):
                if issue_text:
                    logger.warning(f"ISSUE REPORTED by user: {issue_text}")
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("Please describe the issue before submitting.")

        # --- PDF Viewer Display ---
        display_pdf_viewer()


# --- Main Application Logic ---

def aggregate_analysis_results(
    individual_results: List[Tuple[str, str, str]], # List of (title, sub_prompt, analysis_json_str) tuples
    original_filename: str,
    original_prompt: str
    ) -> str: # Returns single aggregated JSON string
    """Merges JSON results from multiple sub-prompt analyses into one."""
    final_analysis = {
        "title": f"Aggregated Analysis for {original_filename}", # Default title
        "analysis_sections": {}
    }
    found_valid_title = False # Tracks if we set a better overall title

    for i, (task_title, sub_prompt, analysis_json_str) in enumerate(individual_results):
        try:
            parsed_result = json.loads(analysis_json_str)
            if not isinstance(parsed_result, dict):
                 logger.warning(f"Sub-analysis result {i} is not a dict, skipping.")
                 final_analysis["analysis_sections"][f"error_aggregation_invalid_format_{i}"] = {
                     "Analysis": f"Result from sub-analysis {i+1} (Prompt: '{sub_prompt[:50]}...') was not in the expected dictionary format.",
                     "Supporting_Phrases": ["No relevant phrase found."],
                     "Context": "Aggregation Error"
                 }
                 continue

            # Extract fields from the simplified schema returned by analyze_document
            analysis_summary = parsed_result.get("analysis_summary", "No analysis provided.")
            supporting_quotes = parsed_result.get("supporting_quotes", ["No relevant phrase found."])
            analysis_context = parsed_result.get("analysis_context", "") # Context from analysis LLM

            # --- Use the AI-generated title to create the section key ---
            # Clean up the AI-generated title to create a valid snake_case key
            section_key = re.sub(r'[^a-zA-Z0-9]', '_', task_title.lower()) # Use task_title
            section_key = re.sub(r'_+', '_', section_key)  # Replace multiple underscores
            section_key = section_key.strip('_')[:60]  # Limit length and remove leading/trailing

            if not section_key: # Fallback if title was empty or only symbols
                section_key = f"sub_prompt_{i+1}"

            # Add a numeric suffix if the key already exists (handles duplicate titles)
            base_key = section_key
            counter = 1
            while section_key in final_analysis["analysis_sections"]:
                section_key = f"{base_key}_{counter}"
                counter += 1

            # Create the section with the proper format expected by the display logic
            # Include the original sub-prompt in the context for clarity
            final_analysis["analysis_sections"][section_key] = {
                "Analysis": analysis_summary,
                "Supporting_Phrases": supporting_quotes,
                "Context": f"AI Title: '{task_title}' | Sub-prompt: '{sub_prompt}'" + (f" | LLM Context: {analysis_context}" if analysis_context else "")
            }

            # Set a better overall title if we haven't yet and this is the first valid analysis
            if not found_valid_title and analysis_summary and "error" not in analysis_summary.lower() and "skipped" not in analysis_summary.lower():
                final_analysis["title"] = f"Analysis of {original_filename}"
                found_valid_title = True

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for sub-analysis result {i} (Prompt: '{sub_prompt[:50]}...'): {e}")
            final_analysis["analysis_sections"][f"error_aggregation_json_decode_{i}"] = {
                "Analysis": f"Failed to parse JSON output from sub-analysis {i+1} (Prompt: '{sub_prompt[:50]}...'). Error: {e}",
                "Supporting_Phrases": ["No relevant phrase found."],
                "Context": "Aggregation Error"
            }
        except Exception as e:
            logger.error(f"Unexpected error aggregating sub-analysis result {i} (Prompt: '{sub_prompt[:50]}...'): {e}", exc_info=True)
            final_analysis["analysis_sections"][f"error_aggregation_unexpected_{i}"] = {
                "Analysis": f"Unexpected error processing result from sub-analysis {i+1} (Prompt: '{sub_prompt[:50]}...'). Error: {e}",
                "Supporting_Phrases": ["No relevant phrase found."],
                "Context": "Aggregation Error"
            }

    # If no valid title was found after aggregation, adjust the default
    if not found_valid_title and not final_analysis["analysis_sections"]:
        final_analysis["title"] = f"Analysis Failed for {original_filename}"
    elif not found_valid_title:
         # Keep the default or maybe indicate issues
         final_analysis["title"] = f"Aggregated Analysis for {original_filename} (Check Sections)"


    return json.dumps(final_analysis, indent=2)


def process_file_wrapper(args):
    """
    Wrapper for processing a single file: decompose prompt, run RAG & analysis per sub-prompt, aggregate results.
    """
    (
        uploaded_file_data,
        filename,
        user_prompt,
        use_advanced_extraction, # Keep if PDFProcessor uses it
    ) = args

    if embedding_model is None:
        logger.error(f"Skipping processing for {filename}: Embedding model not loaded.")
        return {"filename": filename, "error": "Embedding model failed to load.", "annotated_pdf": None, "verification_results": {}, "phrase_locations": {}, "ai_analysis": json.dumps({"error": "Embedding model failed to load."})}

    logger.info(f"Thread {threading.current_thread().name} starting processing for: {filename}")
    start_time = datetime.now()

    try:
        file_extension = Path(filename).suffix.lower()
        processor = None
        original_pdf_bytes_for_annotation = None

        # --- File Handling & Chunk Extraction ---
        if file_extension == ".pdf":
            original_pdf_bytes_for_annotation = uploaded_file_data
            processor = PDFProcessor(uploaded_file_data)
            chunks, _ = processor.extract_structured_text_and_chunks()
        elif file_extension == ".docx":
            word_processor = WordProcessor(uploaded_file_data)
            pdf_bytes = word_processor.convert_to_pdf_bytes()
            if not pdf_bytes: raise ValueError("Failed to convert DOCX to PDF.")
            original_pdf_bytes_for_annotation = pdf_bytes
            processor = PDFProcessor(pdf_bytes) # Process the converted PDF
            chunks, _ = processor.extract_structured_text_and_chunks()
        else:
            # Should not happen due to file uploader limits, but handle defensively
            raise ValueError(f"Unsupported file type: {file_extension}")

        if not chunks:
            logger.warning(f"No chunks extracted for {filename}, cannot proceed with analysis.")
            b64_pdf = base64.b64encode(original_pdf_bytes_for_annotation).decode() if original_pdf_bytes_for_annotation else None
            return {"filename": filename, "error": "No text chunks could be extracted.", "annotated_pdf": b64_pdf, "verification_results": {}, "phrase_locations": {}, "ai_analysis": json.dumps({"error": "Failed to extract text chunks."})}

        analyzer = DocumentAnalyzer()

        # --- Step 1: Decompose Prompt ---
        decomposed_tasks = [] # Will store list of dicts: {'title': ..., 'sub_prompt': ...}
        try:
            decomposed_tasks = run_async(analyzer.decompose_prompt(user_prompt))
            # Check if fallback occurred (returns list with one item and default title)
            if len(decomposed_tasks) == 1 and decomposed_tasks[0]['title'] == "Overall Analysis":
                 logger.info(f"Decomposition failed or resulted in single prompt. Processing original prompt for {filename}.")
                 # Keep the single item in decomposed_tasks
            else:
                 logger.info(f"Decomposed prompt for {filename} into {len(decomposed_tasks)} sub-prompts with titles.")
        except Exception as decomp_err:
            logger.error(f"Prompt decomposition failed for {filename}: {decomp_err}. Processing original prompt as fallback.", exc_info=True)
            decomposed_tasks = [{"title": "Overall Analysis", "sub_prompt": user_prompt}] # Fallback

        # Change structure: List of (title, sub_prompt, analysis_json_str) tuples
        individual_analysis_results: List[Tuple[str, str, str]] = []
        all_relevant_chunk_ids = set() # Use set for automatic deduplication

        # --- Steps 2 & 3: Loop - Targeted RAG & Focused Analysis per Sub-Prompt ---
        for i, task in enumerate(decomposed_tasks):
             sub_start_time = datetime.now()
             task_title = task['title']
             sub_prompt = task['sub_prompt']
             logger.info(f"Processing sub-task {i+1}/{len(decomposed_tasks)} for {filename}: Title='{task_title}', Prompt='{sub_prompt[:60]}...'")

             # Step 2: RAG for this sub-prompt
             relevant_text, relevant_chunk_ids = retrieve_relevant_chunks(
                 sub_prompt, chunks, embedding_model, RAG_TOP_K
             )
             all_relevant_chunk_ids.update(relevant_chunk_ids) # Add retrieved chunks to the set

             # Step 3: AI Analysis for this sub-prompt
             analysis_json_str = run_async(
                 analyzer.analyze_document(relevant_text, filename, sub_prompt)
             )
             # Append title, sub_prompt, and result
             individual_analysis_results.append((task_title, sub_prompt, analysis_json_str))
             sub_elapsed = (datetime.now() - sub_start_time).total_seconds()
             logger.info(f"Sub-task {i+1}/{len(decomposed_tasks)} completed in {sub_elapsed:.2f}s")


        # --- Step 4: Aggregate Results ---
        logger.info(f"Aggregating {len(individual_analysis_results)} analysis results for {filename}.")
        aggregated_ai_analysis_json_str = aggregate_analysis_results(
            individual_analysis_results, filename, user_prompt # Pass the modified list
        )

        # --- Step 5: Verification & Annotation (on aggregated results) ---
        # Verification uses the *original* processor instance with all chunks
        logger.info(f"Verifying phrases from aggregated analysis for {filename}.")
        verification_results, phrase_locations = processor.verify_and_locate_phrases(
            aggregated_ai_analysis_json_str # Use aggregated result
        )

        # Annotation uses the *original* PDF bytes
        logger.info(f"Adding annotations to PDF for {filename}.")
        # Re-initialize PDFProcessor ONLY if original bytes differ from processed bytes (e.g., DOCX conversion)
        # In this structure, original_pdf_bytes_for_annotation holds the correct bytes.
        annotation_processor = PDFProcessor(original_pdf_bytes_for_annotation)
        annotated_pdf_bytes = annotation_processor.add_annotations(phrase_locations)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"Successfully processed {filename} in {total_duration:.2f} seconds.")

        return {
            "filename": filename,
            "annotated_pdf": base64.b64encode(annotated_pdf_bytes).decode() if annotated_pdf_bytes else None,
            "verification_results": verification_results,
            "phrase_locations": phrase_locations,
            "ai_analysis": aggregated_ai_analysis_json_str, # Return aggregated result
            "retrieved_chunk_ids": list(all_relevant_chunk_ids), # Unique list
        }

    except Exception as e:
        logger.error(f"Error in process_file_wrapper for {filename}: {str(e)}", exc_info=True)
        # Try to get original bytes if available for error display
        err_pdf_bytes = None
        if 'original_pdf_bytes_for_annotation' in locals() and original_pdf_bytes_for_annotation:
             err_pdf_bytes = original_pdf_bytes_for_annotation
        elif 'uploaded_file_data' in locals() and file_extension == '.pdf':
             err_pdf_bytes = uploaded_file_data

        b64_err_pdf = base64.b64encode(err_pdf_bytes).decode() if err_pdf_bytes else None
        return {
            "filename": filename,
            "error": f"Processing failed: {str(e)}",
            "annotated_pdf": b64_err_pdf,
            "verification_results": {},
            "phrase_locations": {},
            "ai_analysis": json.dumps({"error": f"Failed to process file: {e}"}),
        }


def display_page():
    """Main function to display the Streamlit page with prompt decomposition."""
    # --- Initialize Session State ---
    if "analysis_results" not in st.session_state: st.session_state.analysis_results = []
    if "show_pdf" not in st.session_state: st.session_state.show_pdf = False
    if "pdf_page" not in st.session_state: st.session_state.pdf_page = 1
    if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None
    if "current_pdf_name" not in st.session_state: st.session_state.current_pdf_name = None
    if "user_prompt" not in st.session_state: st.session_state.user_prompt = ""
    if "use_advanced_extraction" not in st.session_state: st.session_state.use_advanced_extraction = False
    if "last_uploaded_filenames" not in st.session_state: st.session_state.last_uploaded_filenames = set()
    if "uploaded_file_objects" not in st.session_state: st.session_state.uploaded_file_objects = []


    # --- UI Layout ---
    st.markdown(
        "<h1 style='text-align: center;'>SmartDocs Document Analysis (Decomposition Enabled)</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-style: italic;'>Complex prompts are now decomposed into sub-questions for more focused analysis.</p>",
        unsafe_allow_html=True,
    )


    # Check if embedding model loaded successfully
    if embedding_model is None:
        st.error(
            "Embedding model failed to load. Document processing is disabled. "
            "Please check logs and ensure dependencies ('sentence-transformers', 'torch', etc.) are installed correctly."
        )
        return # Stop further UI rendering

    # --- File Upload ---
    uploaded_files = st.file_uploader(
        "Upload PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader_decompose", # Unique key
    )

    current_uploaded_filenames = set(f.name for f in uploaded_files) if uploaded_files else set()
    if current_uploaded_filenames != st.session_state.get('last_uploaded_filenames', set()):
        st.session_state.uploaded_file_objects = uploaded_files
        st.session_state.last_uploaded_filenames = current_uploaded_filenames
        logger.info(f"File selection changed. New files: {st.session_state.last_uploaded_filenames}")
        # Clear results when files change to avoid confusion
        st.session_state.analysis_results = []
        st.session_state.show_pdf = False
        st.session_state.pdf_bytes = None
        st.session_state.current_pdf_name = None
        st.rerun() # Rerun to reflect cleared state


    # --- Analysis Inputs ---
    with st.container(border=False):
        # st.subheader("Ask SmartDocs")
        st.session_state.user_prompt = st.text_area(
            "Analysis Prompt",
            placeholder="Enter your analysis instructions. Multiple questions or tasks will be handled separately (e.g., 'What is the termination clause? What are the liability limits?')",
            height=150,
            key="prompt_input_decompose",
            value=st.session_state.get("user_prompt", ""),
        )

        # Optional: Keep advanced extraction toggle if needed
        # st.session_state.use_advanced_extraction = st.toggle(...)


    # --- Process Button ---
    process_button_disabled = (
        embedding_model is None
        or not st.session_state.get('uploaded_file_objects')
        or not st.session_state.get('user_prompt', '').strip()
    )
    if st.button("Process Documents", type="primary", use_container_width=True, disabled=process_button_disabled):
        files_to_process = st.session_state.get("uploaded_file_objects", [])
        current_user_prompt = st.session_state.get("user_prompt", "")
        current_use_advanced = st.session_state.get("use_advanced_extraction", False) # Read if toggle exists

        if not files_to_process: st.warning("Please upload one or more documents.")
        elif not current_user_prompt.strip(): st.error("Please enter an Analysis Prompt.")
        else:
            st.session_state.analysis_results = []
            st.session_state.show_pdf = False
            st.session_state.pdf_bytes = None
            st.session_state.current_pdf_name = None

            total_files = len(files_to_process)
            overall_start_time = datetime.now()
            # Use a list that maps directly to input files to maintain order
            results_placeholder = [None] * total_files
            file_map = {i: f.name for i, f in enumerate(files_to_process)} # Map index to filename

            process_args = []
            files_read_ok = True
            for i, uploaded_file in enumerate(files_to_process):
                try:
                    file_data = uploaded_file.getvalue()
                    process_args.append(
                        (file_data, uploaded_file.name, current_user_prompt, current_use_advanced)
                    )
                except Exception as read_err:
                    logger.error(f"Failed to read file {uploaded_file.name}: {read_err}", exc_info=True)
                    st.error(f"Failed to read file {uploaded_file.name}. Please re-upload.")
                    results_placeholder[i] = {"filename": uploaded_file.name, "error": f"Failed to read file: {read_err}"}
                    files_read_ok = False

            if files_read_ok and process_args:
                files_to_run_count = len(process_args)
                spinner_msg = (
                    f"Processing {files_to_run_count} document(s)... "
                    "(Decomposing Prompt, Retrieving Chunks per Sub-Prompt, Analyzing)"
                 )
                with st.spinner(spinner_msg, show_time=True):
                    processed_indices = set()

                    def run_task_with_index(item_index: int, args_tuple: tuple):
                        """Wrapper to run task and return index + result."""
                        filename = args_tuple[1]
                        logger.info(f"Thread {threading.current_thread().name} starting task for index {item_index} ({filename})")
                        try:
                            result = process_file_wrapper(args_tuple)
                            logger.info(f"Thread {threading.current_thread().name} finished task for index {item_index} ({filename})")
                            return item_index, result
                        except Exception as thread_err:
                            logger.error(f"Unhandled error in thread task for index {item_index} ({filename}): {thread_err}", exc_info=True)
                            return item_index, {"filename": filename, "error": f"Unhandled thread error: {thread_err}"}

                    # --- Execute Tasks ---
                    try:
                        if ENABLE_PARALLEL and files_to_run_count > 1:
                            logger.info(f"Using ThreadPoolExecutor with {MAX_WORKERS} workers for {files_to_run_count} tasks.")
                            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                                future_map = {
                                    executor.submit(run_task_with_index, i, arg_tuple): i
                                    for i, arg_tuple in enumerate(process_args)
                                }
                                for future in concurrent.futures.as_completed(future_map):
                                    original_index = future_map[future]
                                    processed_indices.add(original_index)
                                    try:
                                        _, result_data = future.result()
                                        results_placeholder[original_index] = result_data
                                    except Exception as exc:
                                        fname = file_map.get(original_index, f"File at index {original_index}")
                                        logger.error(f'Task for index {original_index} ({fname}) failed: {exc}', exc_info=True)
                                        results_placeholder[original_index] = {"filename": fname, "error": f"Task execution failed: {exc}"}
                        else:
                            logger.info(f"Processing {files_to_run_count} task(s) sequentially.")
                            for i, arg_tuple in enumerate(process_args):
                                original_index = i
                                processed_indices.add(original_index)
                                try:
                                    _, result_data = run_task_with_index(original_index, arg_tuple)
                                    results_placeholder[original_index] = result_data
                                except Exception as seq_exc:
                                     fname = file_map.get(original_index, f"File at index {original_index}")
                                     logger.error(f'Sequential task for index {original_index} ({fname}) failed: {seq_exc}', exc_info=True)
                                     results_placeholder[original_index] = {"filename": fname, "error": f"Task execution failed: {seq_exc}"}

                    except Exception as pool_err:
                         logger.error(f"Error during task execution setup/management: {pool_err}", exc_info=True)
                         st.error(f"Error during processing: {pool_err}. Some files may not have been processed.")
                         # Mark remaining unprocessed as errors
                         for i in range(total_files):
                              if i not in processed_indices and results_placeholder[i] is None:
                                   fname = file_map.get(i, f"File at index {i}")
                                   results_placeholder[i] = {"filename": fname, "error": f"Processing cancelled due to execution error: {pool_err}"}


                    # --- Processing Done - Update State ---
                    final_results = [r for r in results_placeholder if r is not None]
                    st.session_state.analysis_results = final_results

                    total_time = (datetime.now() - overall_start_time).total_seconds()
                    success_count = len([r for r in final_results if isinstance(r, dict) and "error" not in r])
                    logger.info(f"Processing batch complete. Processed {success_count}/{total_files} files successfully in {total_time:.2f}s.")

                    # --- Set initial PDF view (first success) ---
                    first_success = next((r for r in final_results if isinstance(r, dict) and "error" not in r), None)
                    if first_success and first_success.get("annotated_pdf"):
                        try:
                            pdf_bytes = base64.b64decode(first_success["annotated_pdf"])
                            update_pdf_view(pdf_bytes=pdf_bytes, page_num=1, filename=first_success.get("filename"))
                        except Exception as decode_err:
                            logger.error(f"Failed to decode/set initial PDF: {decode_err}", exc_info=True)
                            st.error("Failed to load initial PDF view.")
                            st.session_state.show_pdf = False
                    elif first_success:
                         logger.warning("First successful result missing annotated PDF data.")
                         st.warning("Processing complete, but couldn't display the first annotated document.")
                         st.session_state.show_pdf = False
                    else:
                         logger.warning("No successful results found. No initial PDF view shown.")
                         st.session_state.show_pdf = False

            # Rerun to display results / updated PDF view state
            st.rerun()


# --- Display Results Section (Shows after processing button clicked and page reruns) ---
if st.session_state.get("analysis_results"):
    st.divider()
    st.markdown("## Processing Results")

    results_to_display = st.session_state.get("analysis_results", [])
    errors = [r for r in results_to_display if isinstance(r, dict) and "error" in r]
    success_results = [r for r in results_to_display if isinstance(r, dict) and "error" not in r]

    # --- Status Summary ---
    total_processed = len(results_to_display)
    if errors:
        if not success_results: st.error(f"Processing failed for all {total_processed} file(s). See details below.")
        else: st.warning(f"Processing complete for {total_processed} file(s). {len(success_results)} succeeded, {len(errors)} failed.")
    elif success_results: st.success(f"Successfully processed {len(success_results)} file(s).")

    # --- Error Details Expander ---
    if errors:
        with st.expander("⚠️ Processing Errors", expanded=True):
            for error_res in errors:
                st.error(f"**{error_res.get('filename', 'Unknown File')}**: {error_res.get('error', 'Unknown error details.')}")

    # --- Display Successful Analysis ---
    # display_analysis_results now handles the aggregated JSON format
    if success_results:
        # Since aggregation happens per file, we still iterate, but usually expect one success item
        # If multiple files were processed, display_analysis_results will render each one
        display_analysis_results(success_results)
    elif not errors:
        # This case might happen if processing completed but yielded no actionable results (e.g., only info/skipped sections)
        st.warning("Processing finished, but no primary analysis content was generated.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Check model again before displaying page, though initial check should handle most cases
    if embedding_model is not None:
        display_page()
    else:
        # If the model failed, display_page() will show an error,
        # but we can add a log here just in case.
        logger.critical("Application cannot start because the embedding model failed to load.")
        # Avoid st calls here if it might not be initialized