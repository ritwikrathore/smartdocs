# SmartDocs Document Analysis

## Project Description

SmartDocs is a web-based application built with Streamlit designed for intelligent analysis of PDF and DOCX documents. It leverages Large Language Models (LLMs) like Google's Gemini, combined with Retrieval-Augmented Generation (RAG), to provide focused answers to user queries about document content. A key feature is its ability to decompose complex user prompts into smaller, manageable sub-questions, allowing for more granular and accurate analysis of specific document sections and generating citations.

Try it out: https://smartdocs.streamlit.app/

## Key Features

*   **File Upload:** Supports uploading of `.pdf` and `.docx` files.
*   **AI-Powered Analysis:** Uses Google Generative AI models (configurable, e.g., Gemini Flash) to analyze document content based on user prompts.
*   **Prompt Decomposition:** Breaks down complex user questions into multiple sub-prompts for targeted RAG and analysis.
*   **Retrieval-Augmented Generation (RAG):**
    *   Chunks documents into manageable text segments.
    *   Uses sentence transformers (`all-MiniLM-L6-v2` by default) to embed text chunks and user sub-prompts.
    *   Retrieves the most relevant text chunks based on semantic similarity to answer each sub-prompt.
*   **Structured Output:** Generates analysis in a structured JSON format, including the analysis summary and exact supporting quotes from the document.
*   **Phrase Verification:** Uses fuzzy matching (`thefuzz`) to verify if the supporting quotes generated by the AI can be found in the original document text.
*   **PDF Annotation & Viewer:**
    *   Highlights verified supporting phrases directly on a rendered view of the PDF.
    *   Provides an interactive PDF viewer within the Streamlit app to navigate the document and view annotations.
*   **Result Export:** Allows exporting the structured analysis results to Excel (`.xlsx`) and JSON (`.json`) formats.
*   **Configurable:** Settings like AI models, RAG parameters, fuzzy matching threshold, and parallel processing can be configured via environment variables.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ritwikrathore/smartdocs.git
    cd smartdocs
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application uses environment variables for configuration.

1.  **Create a `secrets.toml` file** in the root directory of the project under `.streamlit` folder.
2.  **Add the following variables** to the `secrets.toml` file:

    ```env
    # Required: Your Google AI API Key
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"

    # Optional: Configure processing parameters (defaults shown)
    # MAX_WORKERS=4
    # ENABLE_PARALLEL=true
    # FUZZY_MATCH_THRESHOLD=88
    # RAG_TOP_K=10
    # EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
    # DECOMPOSITION_MODEL_NAME="gemini-1.5-flash"
    # ANALYSIS_MODEL_NAME="gemini-2.0-flash" # Check available model names
    ```
3.  Replace `"YOUR_API_KEY_HERE"` with your actual Google Generative AI API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/).

4.  Create a `config.toml` file in the root directory of the project under `.streamlit` folder. 

    ```toml
    [server]
    folderWatchBlacklist = ["**/torch/**"] # This is to prevent the app from watching the torch library.
    ```

## Usage

1.  Ensure your virtual environment is activated and the `secrets.toml` and `config.toml` files are configured.
2.  Run the Streamlit application:
    ```bash
    streamlit run single_file_app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
4.  Upload PDF or DOCX files using the file uploader.
5.  Enter your analysis prompt in the text area.
6.  Click "Process Documents" to start the analysis.
7.  View the results, explore citations, and interact with the PDF viewer.

## Pending Tasks

1.  Enable chat mode for the application. (Import from old app, enable multi pdf chat, etc.)
2.  Enable Report Issue feature.
3.  Export the analysis results to a PDF file / Word document. (Check existing functionality.)
4.  E̶x̶p̶o̶r̶t̶ ̶A̶n̶n̶o̶t̶a̶t̶e̶d̶ ̶P̶D̶F̶.̶
5.  UI/UX Improvements (PDF Title, App Title, Logos, Container Styling, etc.)
6.  Refactor the code to make it more modular and easier to maintain. (Break down the `single_file_app.py` into smaller functions and classes.)
7.  Bring the Title page to the top, it slides down after analysis.
8.  Fix context content to be more informative.
9.  Add st.badge in citation section.
10. Test multiple pdfs.



Details on how to contribute can be added here (e.g., pull request process, coding standards).

