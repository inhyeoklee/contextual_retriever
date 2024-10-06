import openai
import json
import os
import logging
from config import OPENAI_API_KEY
import sys  # Added import for sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def generate_context(document_text, chunk):
    """Generate context for a given chunk within a document."""
    prompt = f"""<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.5,
            n=1,
        )
        generated_text = response.choices[0].message.content.strip()
        return generated_text
    except Exception as e:
        logging.error(f"Error generating context for chunk: {chunk[:30]}... - {e}")
        return ""

if __name__ == "__main__":
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load the extracted text from the document
        with open(os.path.join(output_dir, 'document_text.txt'), 'r') as f:
            document_text = f.read()
    except FileNotFoundError:
        logging.error("document_text.txt file not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading document text: {e}")
        sys.exit(1)

    try:
        # Load the chunks generated previously
        with open(os.path.join(output_dir, 'chunks.json'), 'r') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        logging.error("chunks.json file not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        sys.exit(1)

    # Generate context for each chunk
    contextualized_chunks = []
    for chunk in chunks:
        context = generate_context(document_text, chunk)
        contextualized_chunk = context + " " + chunk
        contextualized_chunks.append({
            'chunk': chunk,
            'contextualized_chunk': contextualized_chunk
        })

    # Save the contextualized chunks
    with open(os.path.join(output_dir, 'contextualized_chunks.json'), 'w') as f:
        json.dump(contextualized_chunks, f, indent=2)
    logging.info("Contextualized chunks saved.")
