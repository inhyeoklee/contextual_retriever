import os
import sys
import subprocess
import shutil

output_dir = 'output'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

def process_pdf(pdf_path):
    # Step 1: Extract text from PDF
    print(f"Extracting text from {pdf_path}...")
    subprocess.run(['python', 'extract_text.py', pdf_path])

    # Step 2: Chunk the text
    print("Chunking the text...")
    subprocess.run(['python', 'chunk_text.py'])

    # Step 3: Contextualize the chunks
    print("Contextualizing the chunks...")
    subprocess.run(['python', 'chunk_and_contextualize.py'])

    # Step 4: Build the FAISS index
    print("Building the FAISS index...")
    subprocess.run(['python', 'build_index.py'])

def main(directory_path):
    # List all PDF files in the directory
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        process_pdf(pdf_path)

    # Step 5: Interactive querying
    print("The system is ready for queries. Type your questions below.")
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        subprocess.run(['python', 'retrieve.py', query])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <path_to_directory>")
        sys.exit(1)
    directory_path = sys.argv[1]
    main(directory_path)
