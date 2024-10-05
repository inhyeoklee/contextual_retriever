import json
import os

if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Read the extracted text from 'document_text.txt'
    with open(os.path.join(output_dir, 'document_text.txt'), 'r') as f:
        document_text = f.read()

    # Split the text into chunks (you can adjust the chunk size as needed)
    chunk_size = 500  # Define the number of characters per chunk
    chunks = [document_text[i:i + chunk_size] for i in range(0, len(document_text), chunk_size)]

    # Save the chunks to 'chunks.json'
    with open(os.path.join(output_dir, 'chunks.json'), 'w') as f:
        json.dump(chunks, f, indent=2)