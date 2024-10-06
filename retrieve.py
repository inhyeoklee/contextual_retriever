import sys
import json
import faiss
import numpy as np
import os
import voyageai
import pickle
from rank_bm25 import BM25Okapi
from config import Config  # Import Config class
import tiktoken  # Added for token counting
import google.generativeai as genai  # Import Gemini API

# Initialize clients
config = Config()
voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
# Configure Gemini API
api_key = config.GEMINI_API_KEY
genai.configure(api_key=api_key)


def embed_query(query):
    """Embed a query using the Voyage AI client."""
    try:
        response = voyage_client.embed(
            texts=[query],
            model="voyage-3",
            input_type="query"
        )
        return np.array(response.embeddings[0]).astype('float32')
    except Exception as e:
        print(f"Error in embed_query: {e}")
        return None


def embed_texts(texts):
    """Embed a list of texts using the Voyage AI client."""
    try:
        response = voyage_client.embed(
            texts=texts,
            model="voyage-3",
            input_type="document"
        )
        return np.array(response.embeddings).astype('float32')
    except Exception as e:
        print(f"Error in embed_texts: {e}")
        return None


def count_tokens_for_messages(messages, model="gemini-1.5-pro"):
    """Estimate the total number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 4  # Every message requires a fixed number of tokens
    tokens_per_name = -1  # If the role is 'assistant'

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 2  # Accounts for reply assistant tokens
    return num_tokens


def generate_answer(query, chunks):
    """Generate an answer to the query using the provided chunks as context."""
    try:
        if not chunks:
            return "The answer to your question was not found in the provided document."

        # Create initial messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an extremely thoughtful and verbose assistant who can help user extremely well with complicated problems that require sophisticated reasoning. "
                    "You are mathematically enlightened and rely heavily on mathematical and statistical reasoning. You think aloud generally. You use tags strictly as instructed. "
                    "You do not express your own views or beliefs beyond what's strictly necessary to follow the instruction. Your life depends on strictly following the user instruction. "
                    "Follow the user instruction below in a comprehensive and detailed way."
                )
            },
            {
                "role": "user",
                "content": f"Using the information below, answer the following question:\n\nQuestion: {query}\n\nContext:\n"
            }
        ]

        # Increase maximum tokens for the assistant's reply
        max_tokens = 8192  # Allow for longer responses with Gemini
        max_total_tokens = 2097152  # Gemini's maximum context length
        tokens_reserved = max_tokens + 500  # Reserve tokens for reply and buffer
        max_context_tokens = max_total_tokens - tokens_reserved

        # Use tiktoken to count tokens
        encoding = tiktoken.encoding_for_model("gemini-1.5-pro")

        # Add chunks until tokens exceed limit
        context_chunks = []
        total_context_tokens = 0

        for chunk in chunks:
            chunk_tokens = len(encoding.encode(chunk))
            if total_context_tokens + chunk_tokens > max_context_tokens:
                break
            context_chunks.append(chunk)
            total_context_tokens += chunk_tokens

        # Reconstruct the user message with context
        context = "\n\n".join(context_chunks)
        messages[1]['content'] += context + "\n\nPlease provide a detailed answer."

        # Re-count tokens to ensure within limit
        total_tokens = count_tokens_for_messages(messages)
        if total_tokens > max_total_tokens - max_tokens:
            # Reduce context further if necessary
            while total_tokens > max_total_tokens - max_tokens and context_chunks:
                removed_chunk = context_chunks.pop()
                total_context_tokens -= len(encoding.encode(removed_chunk))
                context = "\n\n".join(context_chunks)
                messages[1]['content'] = f"Using the information below, answer the following question:\n\nQuestion: {query}\n\nContext:\n{context}\n\nPlease provide a detailed answer."
                total_tokens = count_tokens_for_messages(messages)

        # Gemini API call
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            messages[1]['content'],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.2
            )
        )
        answer = response.text.strip()
        return answer
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        return "An error occurred while generating the answer."


def retrieve_chunks(query, index, texts, bm25, k=20):
    """Retrieve relevant chunks for a query using both embedding and BM25 methods."""
    try:
        query_vec = embed_query(query)
        if query_vec is None:
            return []
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
        if combined_embeddings is None:
            return combined_results[:k]
        # Normalize embeddings
        combined_embeddings_norm = combined_embeddings / np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
        query_vec_norm = query_vec / np.linalg.norm(query_vec)
        similarity_scores = np.dot(combined_embeddings_norm, query_vec_norm)
        # Sort the results by similarity scores
        reranked_indices = np.argsort(similarity_scores)[::-1][:k]
        reranked_results = [combined_results[idx] for idx in reranked_indices]

        # Log the top results for debugging
        print("Top retrieved chunks:")
        for i, result in enumerate(reranked_results[:5]):
            print(f"Rank {i+1}: {result[:200]}...")

        return reranked_results
    except Exception as e:
        print(f"Error in retrieve_chunks: {e}")
        return []


if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Load indexes and texts once
    try:
        index = faiss.read_index(os.path.join(output_dir, 'faiss_index.index'))
        with open(os.path.join(output_dir, 'texts.json'), 'r', encoding='utf-8') as f:
            texts = json.load(f)
        with open(os.path.join(output_dir, 'bm25_index.pkl'), 'rb') as f:
            bm25 = pickle.load(f)
    except Exception as e:
        print(f"Error loading indexes and texts: {e}")
        sys.exit(1)

    while True:
        try:
            query = input("Enter your query (or type 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting the program.")
                break

            # Retrieve relevant chunks
            chunks = retrieve_chunks(query, index, texts, bm25, k=20)

            # Generate answer
            answer = generate_answer(query, chunks)
            print("\nGenerated Answer:\n")
            print(answer)
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            continue
