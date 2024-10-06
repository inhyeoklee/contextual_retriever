import sys
import logging
import tiktoken
import json
import os
import google.generativeai as genai
from config import Config  # Import the Config class

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load Gemini API key from config
config = Config()
genai.configure(api_key=config.GEMINI_API_KEY)

# Initialize tokenizer
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Define constants
MAX_TOKENS = 1048576  # Updated to match Gemini model's input token limit
CHUNK_SIZE = 7000   # Adjust as needed
OVERLAP = 200      # Adjust as needed
MAX_PROMPT_TOKENS = 1000  # Ensure the prompt stays within token limits


def split_into_chunks(text):
    """Split text into chunks of a specified size with overlap."""
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk = encoder.decode(chunk_tokens)
        chunks.append(chunk)
        start += CHUNK_SIZE - OVERLAP
    return chunks


def generate_context(chunks, index):
    """Generate context for a chunk using adjacent chunks."""
    context_chunks = []
    # Get previous chunk
    if index > 0:
        context_chunks.append(chunks[index - 1])
    # Get next chunk
    if index < len(chunks) - 1:
        context_chunks.append(chunks[index + 1])
    # Combine context chunks
    context_text = '\n'.join(context_chunks)
    prompt = f"""
Here is some context from adjacent chunks:
{context_text}

Here is the chunk we want to situate:
{chunks[index]}

Please provide a short, succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
    # Ensure prompt does not exceed MAX_PROMPT_TOKENS
    prompt_tokens = encoder.encode(prompt)
    if len(prompt_tokens) > MAX_PROMPT_TOKENS:
        # Truncate context_text to fit within limits
        allowed_context_length = MAX_PROMPT_TOKENS - len(encoder.encode(prompt.replace(context_text, '')))
        context_tokens = encoder.encode(context_text)
        truncated_context_tokens = context_tokens[:allowed_context_length]
        truncated_context = encoder.decode(truncated_context_tokens)
        prompt = f"""
Here is some context from adjacent chunks:
{truncated_context}

Here is the chunk we want to situate:
{chunks[index]}

Please provide a short, succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
    try:
        # Call Gemini API
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=MAX_PROMPT_TOKENS,
                temperature=0.2
            )
        )
        # Check if response is blocked
        if not response.candidates:
            raise ValueError("Prompt was blocked. Please review the content.")
        # Access response using dot notation
        context = response.text.strip()
        return context
    except Exception as e:
        logging.error(f"Error generating context for chunk {index}: {e}")
        return ""


def process_document(text, output_file):
    """Process a document by splitting it into chunks and generating context for each chunk."""
    logging.info("Splitting text into chunks...")
    chunks = split_into_chunks(text)
    contextualized_chunks = []
    for idx in range(len(chunks)):
        logging.info(f"Processing chunk {idx+1}/{len(chunks)}")
        context = generate_context(chunks, idx)
        contextualized_chunk = context + "\n" + chunks[idx]
        contextualized_chunks.append({'contextualized_chunk': contextualized_chunk})
    # Save the contextualized chunks to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contextualized_chunks, f)
    logging.info(f"Contextualized chunks saved to {output_file}.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python chunk_and_contextualize.py <input_text_file> <output_chunks_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    process_document(text, output_file)
