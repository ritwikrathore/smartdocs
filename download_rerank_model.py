# download_rerank_model.py
from sentence_transformers import CrossEncoder
import os

    # --- Configuration ---
    # <<< IMPORTANT: Replace this with the actual model name used in your app >>>
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
    # <<< This is the directory where the model will be saved >>>
    # <<< Place this directory inside your project structure >>>
SAVE_PATH = "./reranking_model_local"
    # ---

if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Created directory: {SAVE_PATH}")
else:
        print(f"Directory already exists: {SAVE_PATH}")

print(f"Downloading and saving model '{MODEL_NAME}' to '{SAVE_PATH}'...")
try:
        model = CrossEncoder(MODEL_NAME)
        model.save(SAVE_PATH)
        print(f"Model '{MODEL_NAME}' saved successfully to '{SAVE_PATH}'")
except Exception as e:
        print(f"Error downloading or saving model: {e}")
