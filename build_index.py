import sys
import json
import faiss
import numpy as np
import os
import voyageai
from config import Config
from rank_bm25 import BM25Okapi
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = Config()

# Initialize Voyage AI client
client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

def embed_text(texts):
    """Embed a list of texts using the Voyage AI client."""
    embeddings = []
    for text in texts:
        try:
            response = client.embed(
                texts=[text],
                model="voyage-3",
                input_type="document"
            )
            embeddings.extend(response.embeddings)
        except Exception as e:
            logging.error(f"Error embedding text: {text[:30]}... - {e}")
    return np.array(embeddings).astype('float32')

def build_bm25_index(texts):
    """Build a BM25 index from a list of texts."""
    tokenized_texts = [text.split(" ") for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    return bm25

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_index.py <chunks_file>")
        sys.exit(1)

    chunks_file = sys.argv[1]

    # Ensure the output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Load existing texts and indices if they exist
    existing_texts = []
    if os.path.exists(os.path.join(output_dir, 'texts.json')):
        with open(os.path.join(output_dir, 'texts.json'), 'r', encoding='utf-8') as f:
            existing_texts = json.load(f)

    existing_embeddings = None
    if os.path.exists(os.path.join(output_dir, 'faiss_index.index')):
        existing_index = faiss.read_index(os.path.join(output_dir, 'faiss_index.index'))
        existing_embeddings = existing_index.reconstruct_n(0, existing_index.ntotal)

    existing_bm25 = None
    if os.path.exists(os.path.join(output_dir, 'bm25_index.pkl')):
        with open(os.path.join(output_dir, 'bm25_index.pkl'), 'rb') as f:
            existing_bm25 = pickle.load(f)

    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        new_texts = [chunk['contextualized_chunk'] for chunk in chunks_data]
    except FileNotFoundError:
        logging.error(f"{chunks_file} file not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        sys.exit(1)

    # Combine new texts with existing texts
    all_texts = existing_texts + new_texts

    # Embed new texts
    logging.info("Embedding texts...")
    new_embeddings = embed_text(new_texts)

    # Combine new embeddings with existing embeddings
    if existing_embeddings is not None:
        all_embeddings = np.vstack((existing_embeddings, new_embeddings))
    else:
        all_embeddings = new_embeddings

    # Build FAISS index
    logging.info("Building FAISS index...")
    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)

    # Save the FAISS index
    faiss.write_index(index, os.path.join(output_dir, 'faiss_index.index'))
    logging.info("FAISS index saved.")

    # Build BM25 index
    logging.info("Building BM25 index...")
    all_bm25 = build_bm25_index(all_texts)

    # Save the BM25 index
    with open(os.path.join(output_dir, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(all_bm25, f)
    logging.info("BM25 index saved.")

    # Save the texts for reference
    with open(os.path.join(output_dir, 'texts.json'), 'w', encoding='utf-8') as f:
        json.dump(all_texts, f)
    logging.info("Texts saved.")

    logging.info("Index built successfully.")