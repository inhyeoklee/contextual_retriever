import json
import faiss
import numpy as np
import os
from openai import OpenAI
import config
import sys

client = OpenAI(api_key=config.OPENAI_API_KEY)

def embed_query(query):
    response = client.embeddings.create(
        input=query,
        model='text-embedding-ada-002'
    )
    return np.array(response.data[0].embedding).astype('float32')

def retrieve_chunks(query, index, chunks_meta, k=5):
    query_vec = embed_query(query)
    D, I = index.search(np.array([query_vec]), k)
    results = []
    for idx in I[0]:
        results.append(chunks_meta[idx])
    return results

def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
        {"role": "user", "content": f"""Using the information below, answer the following question:

Question: {query}

Context:
{context}

Please provide a detailed answer."""}
    ]
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = input("Enter your query: ")
    index = faiss.read_index(os.path.join(output_dir, 'chunks.index'))
    with open(os.path.join(output_dir, 'chunks_meta.json'), 'r') as f:
        chunks_meta = json.load(f)
    chunks = retrieve_chunks(query, index, chunks_meta, k=5)
    answer = generate_answer(query, chunks)
    print(answer)