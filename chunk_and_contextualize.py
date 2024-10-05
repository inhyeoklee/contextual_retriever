import openai
import json
import os
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def generate_context(document_text, chunk):
    prompt = f"""Provide context for the following chunk based on the document:

Document:
{document_text}

Chunk:
{chunk}

Context:"""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    generated_text = response.choices[0].message.content
    return generated_text.strip()

if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Load the extracted text from the document
    with open(os.path.join(output_dir, 'document_text.txt'), 'r') as f:
        document_text = f.read()

    # Load the chunks generated previously
    with open(os.path.join(output_dir, 'chunks.json'), 'r') as f:
        chunks = json.load(f)

    # Generate context for each chunk
    contextualized_chunks = []
    for chunk in chunks:
        context = generate_context(document_text, chunk)
        contextualized_chunks.append({
            'chunk': chunk,
            'context': context
        })

    # Save the contextualized chunks
    with open(os.path.join(output_dir, 'contextualized_chunks.json'), 'w') as f:
        json.dump(contextualized_chunks, f, indent=2)