# Contextual Retriever

![Puppy](https://github.com/user-attachments/assets/2cac998b-fd40-4a77-b83a-356dd86679ac)

This project implements a contextual retriever using Voyage AI embeddings for document retrieval and reranking. Inspired by Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) article, we rerank using the Voyage AI model and generate answers using the Gemini API.

## Overview

**Contextual Retriever** enhances Retrieval-Augmented Generation (RAG) systems by improving retrieval accuracy when dealing with large knowledge bases. It addresses the loss of context that occurs when documents are split into smaller chunks for processing.

By employing **Contextual Embeddings** and **Contextual BM25**, along with **prompt caching**, the system enriches each chunk with additional context, significantly improving retrieval performance.

## Features

- **Token-Based Chunking with Overlap**: Splits documents into token-based chunks with overlapping tokens to maintain context continuity.
- **Context Generation with Prompt Caching**: Utilizes a language model to generate context for each chunk, reducing costs and improving efficiency.
- **Similarity Search using FAISS**: Builds a FAISS index for efficient similarity search and retrieval.

## API Usage

### Voyage AI API
- **Purpose**: Embedding text and queries.
- **Usage**:
  - **`build_index.py`**: Embeds text data to create embeddings for indexing.
  - **`run_pipeline.py`**: Embeds queries and documents for retrieval tasks.
  - **`retrieve.py`**: Embeds queries and documents to facilitate retrieval and reranking.

### Gemini API
- **Purpose**: Generating context and answers using a generative model.
- **Usage**:
  - **`chunk_and_contextualize.py`**: Generates context for text chunks using adjacent chunks.
  - **`run_pipeline.py`**: Generates answers to queries using retrieved chunks as context.
  - **`retrieve.py`**: Generates detailed answers to user queries using the provided context.

### FAISS
- **Purpose**: Building and querying similarity indexes.
- **Usage**:
  - **`build_index.py`**: Builds a FAISS index for efficient similarity search.
  - **`run_pipeline.py`**: Uses FAISS for retrieving relevant document chunks.
  - **`retrieve.py`**: Retrieves relevant chunks using FAISS index.

### BM25
- **Purpose**: Text retrieval using BM25 algorithm.
- **Usage**:
  - **`build_index.py`**: Builds a BM25 index for text retrieval.
  - **`run_pipeline.py`**: Uses BM25 to retrieve relevant document chunks.
  - **`retrieve.py`**: Retrieves relevant chunks using BM25.

## Installation

### Prerequisites

- Python 3.7 or higher
- Voyage AI API key
- Gemini API key
- Required Python packages (listed in `requirements.txt`)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/contextual_retriever.git
   cd contextual_retriever
   ```

2. **Create a Virtual Environment (Optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

3. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**

   Set your Voyage AI and Gemini API keys in your environment variables:

   ```bash
   export VOYAGE_API_KEY='your-voyage-api-key'
   export GEMINI_API_KEY='your-gemini-api-key'
   ```

## Usage

Run the pipeline script with the path to your documents:

```bash
python run_pipeline.py /path/to/your/documents
```

This will process your documents and create an output directory containing the results.

## Example

Suppose you have a directory of PDF files at `/path/to/your/documents`. To process these documents:

```bash
python run_pipeline.py /path/to/your/documents
```

## Notes

### Prompt Caching

- Implemented in `chunk_and_contextualize.py`.
- Reduces the number of tokens sent to the language model, lowering costs and improving efficiency.
- The system prompt containing the entire document is cached and reused for generating context for each chunk.

## Dependencies

- [FAISS](https://faiss.ai/) for building the similarity index.

## References

- [Anthropic's Contextual Retrieval Article](https://www.anthropic.com/news/contextual-retrieval)
- [FAISS Documentation](https://faiss.ai/)

## Acknowledgments

- Inspired by Anthropic's research on Contextual Retrieval.
- Thanks to the open-source community for the tools and libraries used in this project.

## Code Summary

### extract_text.py
- `extract_text(pdf_path, output_file)`: Extracts text from a PDF file and saves it to a specified output file.

### build_index.py
- `embed_text(texts)`: Embeds a list of texts using the Voyage AI client.
- `build_bm25_index(texts)`: Builds a BM25 index from a list of texts.

### run_pipeline.py
- `process_pdf(pdf_path)`: Processes a PDF file by extracting text, chunking, and building an index.
- `embed_query(query)`: Embeds a query using the Voyage AI client.
- `embed_texts(texts)`: Embeds a list of texts using the Voyage AI client.
- `retrieve_chunks(query, index, texts, bm25, k=20)`: Retrieves relevant chunks for a query using both embedding and BM25 methods.
- `generate_answer(query, chunks)`: Generates an answer to the query using the provided chunks as context.
- `retrieve_and_answer()`: Retrieves relevant chunks and generates an answer for a user query.
- `main(input_path)`: Main function to process PDFs and retrieve answers.

### chunk_and_contextualize.py
- `split_into_chunks(text)`: Splits text into chunks of a specified size with overlap.
- `generate_context(chunks, index)`: Generates context for a chunk using adjacent chunks.
- `process_document(text, output_file)`: Processes a document by splitting it into chunks and generating context for each chunk.

### config.py
- `Config`: Configuration class to manage API keys and other settings.
- `validate_keys()`: Validates that all necessary API keys are set.

### retrieve.py
- `embed_query(query)`: Embeds a query using the Voyage AI client.
- `embed_texts(texts)`: Embeds a list of texts using the Voyage AI client.
- `count_tokens_for_messages(messages, model="gemini-1.5-pro")`: Estimates the total number of tokens used by a list of messages.
- `generate_answer(query, chunks)`: Generates an answer to the query using the provided chunks as context.
- `retrieve_chunks(query, index, texts, bm25, k=20)`: Retrieves relevant chunks for a query using both embedding and BM25 methods.
