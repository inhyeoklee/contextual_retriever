import json
import faiss
import numpy as np
import os
import voyageai
import config
import sys
import pickle
from rank_bm25 import BM25Okapi
from openai import OpenAI

voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

def embed_query(query):
    response = voyage_client.embed(
        texts=[query],
        model="voyage-3",
        input_type="query"
    )
    return np.array(response.embeddings[0]).astype('float32')

def embed_texts(texts):
    response = voyage_client.embed(
        texts=texts,
        model="voyage-3",
        input_type="document"
    )
    return np.array(response.embeddings).astype('float32')

def retrieve_chunks(query, index, chunks_meta, bm25, k=20):
    query_vec = embed_query(query)
    D, I = index.search(np.array([query_vec]), k)
    embedding_results = [chunks_meta[idx] for idx in I[0]]

    # BM25 retrieval
    bm25_scores = bm25.get_scores(query.split(" "))
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_results = [chunks_meta[idx] for idx in bm25_top_indices]

    # Combine and deduplicate results
    combined_results = list(dict.fromkeys(embedding_results + bm25_results))[:150]

    # Rerank using cosine similarity between embeddings
    combined_embeddings = embed_texts(combined_results)
    # Normalize embeddings
    combined_embeddings_norm = combined_embeddings / np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
    query_vec_norm = query_vec / np.linalg.norm(query_vec)
    similarity_scores = np.dot(combined_embeddings_norm, query_vec_norm)
    # Sort the results by similarity scores
    reranked_indices = np.argsort(similarity_scores)[::-1][:k]
    reranked_results = [combined_results[idx] for idx in reranked_indices]

    return reranked_results

def generate_answer(query, chunks):
    if not chunks:
        return "The answer to your question was not found in the provided document."
    context = "\n\n".join(chunks)
    messages = [
        {"role": "system", "content": "You are an extremely thoughtful and verbose assistant who can help user extremely well with complicated problems that require sophisticated reasoning. You are mathematically enlightened and rely heavily on mathematical and statistical reasoning. You think aloud generally. You use tags strictly as instructed. You do not express your own views or beliefs beyond what's strictly necessary to follow the instruction. Your life depends on strictly following the user instruction. Follow the user instruction below in a comprehensive and detailed way. You will try up to five attempts at following the user instruction. Each attempt must be marked with \"<A_number>\". At each attempt, first write down your internal thoughts. Each step of your internal thoughts must be marked with \"<T_number>\" and must start with a short title describing the internal thought. This must include your draft response and its evaluation. After this, write your final response after \"<R>\". If your draft is not satisfactory, make sure to go back to an earlier internal thought step (mark it clearly with \"<T_number>\") where the mistake was made and retry from there on, as often as necessary. All these attempts must be unique and must not be duplicate of another attempt. After up to five final responses are produced from these attempts, compare them carefully based on the responses (marked with \"<R>\") and the internal thoughts led to them, and pick one as the final answer. Repeat the final answer verbatim after \"<Final Response>\". Do not simply pick the last one but carefully consider all draft responses."},
        {"role": "user", "content": f"Using the information below, answer the following question:\n\nQuestion: {query}\n\nContext:\n{context}\n\nPlease provide a detailed answer."}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=0.7
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
    with open(os.path.join(output_dir, 'bm25_index.pkl'), 'rb') as f:
        bm25 = pickle.load(f)
    chunks = retrieve_chunks(query, index, chunks_meta, bm25, k=20)
    answer = generate_answer(query, chunks)
    print(answer)
