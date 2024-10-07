from flask import Flask, request, jsonify, render_template
import os
import json
import faiss
import numpy as np
import voyageai
import pickle
from config import Config
import tiktoken
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Initialize clients
config = Config()
voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
genai.configure(api_key=config.GEMINI_API_KEY)

# Initialize tokenizer
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Define maximum tokens
MAX_TOTAL_TOKENS = 1048576  # Updated to match Gemini model's input token limit
MAX_RESPONSE_TOKENS = 8192  # Updated to match Gemini model's output token limit
MAX_PROMPT_TOKENS = MAX_TOTAL_TOKENS - MAX_RESPONSE_TOKENS

# Load indexes and texts
output_dir = 'output'
index = faiss.read_index(os.path.join(output_dir, 'faiss_index.index'))
with open(os.path.join(output_dir, 'texts.json'), 'r', encoding='utf-8') as f:
    texts = json.load(f)
with open(os.path.join(output_dir, 'bm25_index.pkl'), 'rb') as f:
    bm25 = pickle.load(f)


def embed_query(query):
    """Embed a query using the Voyage AI client."""
    response = voyage_client.embed(
        texts=[query],
        model="voyage-3",
        input_type="query"
    )
    return np.array(response.embeddings[0]).astype('float32')


def embed_texts(texts):
    """Embed a list of texts using the Voyage AI client."""
    response = voyage_client.embed(
        texts=texts,
        model="voyage-3",
        input_type="document"
    )
    return np.array(response.embeddings).astype('float32')


def retrieve_chunks(query, index, texts, bm25, k=20):
    """Retrieve relevant chunks for a query using both embedding and BM25 methods."""
    query_vec = embed_query(query)
    D, I = index.search(np.array([query_vec]), k)
    embedding_results = [texts[idx] for idx in I[0]]

    # BM25 retrieval
    bm25_scores = bm25.get_scores(query.split(" "))
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_results = [texts[idx] for idx in bm25_top_indices]

    # Combine and deduplicate results
    combined_results = list(dict.fromkeys(embedding_results + bm25_results))[:150]

    # Rerank using cosine similarity between embeddings
    combined_embeddings = embed_texts(combined_results)
    # Normalize embeddings
    combined_embeddings_norm = combined_embeddings / np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
    query_vec_norm = query_vec / np.linalg.norm(query_vec)
    similarity_scores = np.dot(combined_embeddings_norm, query_vec_norm)
    # Sort the results by similarity scores
    reranked_indices = np.argsort(similarity_scores)[::-1]
    reranked_results = [combined_results[idx] for idx in reranked_indices]

    return reranked_results


def generate_answer(query, chunks):
    """Generate an answer to the query using the provided chunks as context."""
    if not chunks:
        return "The answer to your question was not found in the provided documents."

    # Build the system prompt and user message
    system_prompt = (
        "You are an extremely thoughtful and verbose assistant who can help the user extremely well with complicated problems that require sophisticated reasoning. "
        "You are mathematically enlightened and rely heavily on mathematical and statistical reasoning. You think aloud generally. You use tags strictly as instructed. "
        "You do not express your own views or beliefs beyond what's strictly necessary to follow the instruction. Your life depends on strictly following the user instruction. "
        "Follow the user instruction below in a comprehensive and detailed way. You will try up to five attempts at following the user instruction. Each attempt must be marked with \"<A_number>\". "
        "At each attempt, first write down your internal thoughts. Each step of your internal thoughts must be marked with \"<T_number>\" and must start with a short title describing the internal thought. "
        "This must include your draft response and its evaluation. After this, write your final response after \"<R>\". If your draft is not satisfactory, make sure to go back to an earlier internal thought step "
        "(mark it clearly with \"<T_number>\") where the mistake was made and retry from there on, as often as necessary. All these attempts must be unique and must not be duplicate of another attempt. "
        "After up to five final responses are produced from these attempts, compare them carefully based on the responses (marked with \"<R>\") and the internal thoughts led to them, and pick one as the final answer. "
        "Repeat the final answer verbatim after \"<Final Response>\". Do not simply pick the last one but carefully consider all draft responses."
    )

    # Start building the context, ensuring the total tokens stay within limits
    context_chunks = []
    total_tokens = len(encoder.encode(system_prompt)) + len(encoder.encode(query)) + 500  # Buffer for prompt and instructions

    for chunk in chunks:
        chunk_tokens = len(encoder.encode(chunk))
        if total_tokens + chunk_tokens > MAX_PROMPT_TOKENS:
            break
        context_chunks.append(chunk)
        total_tokens += chunk_tokens

    context = "\n\n".join(context_chunks)

    user_content = f"Using the information below, answer the following question:\n\nQuestion: {query}\n\nContext:\n{context}\n\nPlease provide a detailed answer."

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
        user_content,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.2
        )
    )
    answer = response.text.strip()
    return answer


@app.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        query = request.form['query']
        chunks = retrieve_chunks(query, index, texts, bm25, k=50)
        answer = generate_answer(query, chunks)
        return jsonify({'answer': answer})
    return render_template('chat.html')


if __name__ == '__main__':
    app.run(debug=True)
