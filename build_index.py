import json
import faiss
import numpy as np
import os
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def embed_text(texts):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model='text-embedding-3-large'
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings).astype('float32')

if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'chunks.json'), 'r') as f:
        chunks = json.load(f)
    texts = chunks
    embeddings = embed_text(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, 'chunks.index'))
    # Save the texts as metadata
    with open(os.path.join(output_dir, 'chunks_meta.json'), 'w') as f:
        json.dump(texts, f)