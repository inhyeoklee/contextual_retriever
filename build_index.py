import json
import faiss
import numpy as np
import os
import voyageai
import config
from rank_bm25 import BM25Okapi
import logging
import pickle
import sys  # Added import for sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

def embed_text(texts):
    """Embed a list of texts using the VoyageAI client."""
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
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(os.path.join(output_dir, 'contextualized_chunks.json'), 'r') as f:
            chunks_data = json.load(f)
        texts = [chunk['contextualized_chunk'] for chunk in chunks_data]
    except FileNotFoundError:
        logging.error("contextualized_chunks.json file not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        sys.exit(1)

    # Embed texts and build FAISS index
    embeddings = embed_text(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, 'chunks.index'))
    logging.info("FAISS index built and saved.")

    # Build BM25 index
    bm25 = build_bm25_index(texts)
    with open(os.path.join(output_dir, 'bm25_index.pkl'), 'wb') as f:
        pickle.dump(bm25, f)
    logging.info("BM25 index built and saved.")

    # Save the texts as metadata
    with open(os.path.join(output_dir, 'chunks_meta.json'), 'w') as f:
        json.dump(texts, f)
    logging.info("Metadata saved.")
