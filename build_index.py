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

    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        texts = [chunk['contextualized_chunk'] for chunk in chunks_data]
    except FileNotFoundError:
        logging.error(f"{chunks_file} file not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        sys.exit(1)

    # Proceed to build the index
    logging.info("Embedding texts...")
    embeddings = embed_text(texts)

    # Build FAISS index
    logging.info("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save the FAISS index
    faiss.write_index(index, os.path.join(output_dir, 'faiss_index.index'))
    logging.info("FAISS index saved.")

    # Build BM25 index
    logging.info("Building BM25 index...")
    bm25 = build_bm25_index(texts)

    # Save the BM25 index
    with open(os.path.join(output_dir, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(bm25, f)
    logging.info("BM25 index saved.")

    # Save the texts for reference
    with open(os.path.join(output_dir, 'texts.json'), 'w', encoding='utf-8') as f:
        json.dump(texts, f)
    logging.info("Texts saved.")

    logging.info("Index built successfully.")
