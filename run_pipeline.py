import sys
import os
import subprocess
import json
import faiss
import numpy as np
import voyageai
import pickle
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from config import Config
import tiktoken
import logging
import tkinter as tk
from tkinter import scrolledtext

# Initialize clients
config = Config()
voyage_client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
genai.configure(api_key=config.GEMINI_API_KEY)

# Initialize tokenizer
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Define maximum tokens
MAX_TOTAL_TOKENS = 2097152  # Updated to allow for more detailed responses
MAX_RESPONSE_TOKENS = 16384  # Increased to allow for longer answers
MAX_PROMPT_TOKENS = MAX_TOTAL_TOKENS - MAX_RESPONSE_TOKENS

# Global variables for index and texts
index = None
texts = None
bm25 = None

# Set up logging
logging.basicConfig(level=logging.INFO)


def process_pdf(pdf_path):
    """Process a PDF file by extracting text, chunking, and building an index."""
    print(f"Processing {pdf_path}...")

    # Sanitize filename to replace spaces and special characters
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_base_name = base_name.replace(' ', '_').replace("'", '').replace('"', '')

    extracted_text_file = os.path.join('output', f"{safe_base_name}_extracted.txt")
    chunks_file = os.path.join('output', f"{safe_base_name}_chunks.json")

    # Ensure the 'output' directory exists
    os.makedirs('output', exist_ok=True)

    try:
        # Extract text from the PDF
        print(f"Extracting text from {pdf_path}...")
        subprocess.run(['python', 'extract_text.py', pdf_path, extracted_text_file], check=True)

        # Chunk and contextualize the text
        print("Chunking and contextualizing the text...")
        subprocess.run(['python', 'chunk_and_contextualize.py', extracted_text_file, chunks_file], check=True)

        # Build the index
        print("Building the index...")
        subprocess.run(['python', 'build_index.py', chunks_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during processing: {e}")


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

    # Use Voyage AI reranker
    reranking = voyage_client.rerank(query, combined_results, model="rerank-2")
    reranked_results = [result.document for result in reranking.results]

    return reranked_results


def generate_answer(query, chunks):
    """Generate an answer to the query using the provided chunks as context."""
    if not chunks:
        return "The answer to your question was not found in the provided documents."

    # Detailed system prompt
    system_prompt = "You are an extremely thoughtful and verbose assistant who can help user extremely well with complicated problems that require sophisticated reasoning. You are mathematically enlightened and rely heavily on mathematical and statistical reasoning. You think aloud generally. You use tags strictly as instructed. You do not express your own views or beliefs beyond what's strictly necessary to follow the instruction. Your life depends on strictly following the user instruction."

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

    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            user_content,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=MAX_RESPONSE_TOKENS,
                temperature=0.5  # Adjusted for more creative responses
            )
        )

        if not response.candidates:
            logging.warning(f"Prompt blocked or no response generated for query: {query}")
            return "The prompt was blocked or no response was generated. Please try a different query."

        answer = response.text.strip()
        return answer
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."


def start_chat_interface():
    """Start a Tkinter-based chat interface."""
    def send_query(event=None):
        query = query_entry.get()
        if query.strip() == '':
            return

        chat_display.insert(tk.END, f"You: {query}\n", "user")
        query_entry.delete(0, tk.END)

        # Retrieve relevant chunks
        chunks = retrieve_chunks(query, index, texts, bm25, k=20)  # Reduced k from 50 to 20

        # Generate answer
        answer = generate_answer(query, chunks)
        chat_display.insert(tk.END, f"Bot: {answer}\n", "bot")

    root = tk.Tk()
    root.title("Interactive Chatbot")
    root.geometry("800x600")
    root.configure(bg="#f5f5f5")
    root.resizable(True, True)

    chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("San Francisco", 14), bg="#ffffff", fg="#333333", relief=tk.FLAT)
    chat_display.tag_configure("user", foreground="#007aff")
    chat_display.tag_configure("bot", foreground="#34c759")
    chat_display.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

    query_entry = tk.Entry(root, width=50, font=("San Francisco", 14), bg="#ffffff", fg="#333333", relief=tk.FLAT)
    query_entry.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.X, expand=True)
    query_entry.bind("<Return>", send_query)

    send_button = tk.Button(root, text="Send", command=send_query, font=("San Francisco", 14, "bold"), bg="#e0e0e0", fg="#333333", relief=tk.FLAT)
    send_button.pack(side=tk.LEFT, padx=20, pady=20)

    root.mainloop()


def main(input_path):
    global index, texts, bm25

    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        # Process single PDF file
        process_pdf(input_path)
    elif os.path.isdir(input_path):
        # Process all PDF files in the directory
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_file = os.path.join(root, file)
                    process_pdf(pdf_file)
    else:
        print(f"Invalid path or no PDF files found at: {input_path}")
        sys.exit(1)

    # Load indexes and texts
    try:
        index = faiss.read_index(os.path.join('output', 'faiss_index.index'))
        with open(os.path.join('output', 'texts.json'), 'r', encoding='utf-8') as f:
            texts = json.load(f)
        with open(os.path.join('output', 'bm25_index.pkl'), 'rb') as f:
            bm25 = pickle.load(f)
    except Exception as e:
        print(f"Error loading indexes and texts: {e}")
        sys.exit(1)

    # Start the chat interface
    start_chat_interface()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <pdf_path_or_directory>")
        sys.exit(1)
    input_path = sys.argv[1]
    main(input_path)
