# Contextual Retriever - Local Chatbot with Enhanced RAG

![Puppy](https://github.com/user-attachments/assets/2cac998b-fd40-4a77-b83a-356dd86679ac)

This project implements a contextual retriever using Voyage AI embeddings for document retrieval and reranking. Inspired by Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) research, this tool leverages advanced AI models to enhance retrieval accuracy and generate comprehensive responses.

## Overview

Contextual Retriever is designed to improve Retrieval-Augmented Generation (RAG) systems by addressing the loss of context when documents are split into smaller chunks. By enriching each chunk with additional context, the system significantly enhances retrieval performance, providing users with relevant and comprehensive answers.

## Features

- **Intuitive Chatbot GUI**: A Tkinter-based interface for intuitive querying and response display, allowing users to interact with the system seamlessly.
- **Token-Based Chunking with Overlap**: Documents are split into token-based chunks with overlapping tokens to maintain context continuity, ensuring that no critical information is lost during processing.
- **Context Generation with Prompt Caching**: Utilizes a language model to generate context for each chunk, optimizing efficiency and reducing computational costs by caching prompts.
- **Similarity Search using FAISS**: Builds a FAISS index for efficient similarity search and retrieval, enabling fast and accurate document retrieval.
- **Voyage AI Reranker**: Refines the relevancy of candidate documents using the Voyage AI reranker, which enhances the precision of retrieval results by re-evaluating document relevance.

## Methodology

### Voyage AI API
- **Purpose**: Embedding text and queries for precise retrieval.
- **Integration**:
  - **`build_index.py`**: Utilizes the Voyage AI client to create embeddings for indexing, facilitating efficient document retrieval.
  - **`run_pipeline.py`**: Embeds queries and documents to support retrieval tasks, ensuring accurate and relevant results.
  - **`retrieve.py`**: Employs embeddings to facilitate retrieval and reranking, enhancing the overall retrieval process.

### Gemini API
- **Purpose**: Crafting context and generating answers with a generative model.
- **Integration**:
  - **`chunk_and_contextualize.py`**: Develops context for text chunks by leveraging adjacent chunks, improving the coherence and relevance of generated responses.
  - **`run_pipeline.py`**: Generates comprehensive answers using retrieved chunks as context, providing detailed and insightful responses to user queries.
  - **`retrieve.py`**: Delivers detailed responses by generating answers based on the provided context, ensuring that user queries are addressed thoroughly.

### FAISS
- **Purpose**: Efficient similarity indexing and querying.
- **Integration**:
  - **`build_index.py`**: Constructs FAISS indexes to enable efficient similarity search, supporting fast and accurate document retrieval.
  - **`run_pipeline.py`**: Utilizes FAISS for retrieving relevant document chunks, enhancing the speed and accuracy of the retrieval process.
  - **`retrieve.py`**: Executes retrieval using FAISS indexes, ensuring that the most relevant documents are identified and presented.

### BM25
- **Purpose**: Robust text retrieval with BM25 algorithm.
- **Integration**:
  - **`build_index.py`**: Establishes BM25 indexes for text retrieval, supporting robust and reliable document retrieval.
  - **`run_pipeline.py`**: Employs BM25 to retrieve relevant document chunks, enhancing the precision of the retrieval process.
  - **`retrieve.py`**: Conducts retrieval using BM25, ensuring that relevant documents are accurately identified and ranked.

### Reranking Strategy
- **Purpose**: Refine the relevancy of candidate documents using Voyage AI reranker.
- **Integration**:
  - **`run_pipeline.py`**: Uses `voyageai.Client.rerank()` to rerank candidate documents based on their relevance to the query, ensuring that the most relevant documents are prioritized and presented to the user.

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

Run the pipeline script with the path to your documents, which can be either a folder or a file location:

```bash
python run_pipeline.py /path/to/your/documents
```

This will process your documents and open a chat window for querying.

## Notes

### Prompt Caching

- Implemented in `chunk_and_contextualize.py`.
- Reduces the number of tokens sent to the language model, lowering costs and improving efficiency.
- The system prompt containing the entire document is cached and reused for generating context for each chunk.

## References

- [Anthropic's Contextual Retrieval Article](https://www.anthropic.com/news/contextual-retrieval)

## Acknowledgments

- Inspired by Anthropic's research on Contextual Retrieval.
- Thanks to the open-source community for the tools and libraries used in this project.

## Code Summary

### extract_text.py
- `extract_text(pdf_path, output_file)`: Extracts text from a PDF file and saves it to a specified output file.

### chunk_and_contextualize.py
- `split_into_chunks(text)`: Splits text into chunks of a specified size with overlap.
- `generate_context(chunks, index)`: Generates context for a chunk using adjacent chunks.
- `process_document(text, output_file)`: Processes a document by splitting it into chunks and generating context for each chunk.

### build_index.py
- `embed_text(texts)`: Embeds a list of texts using the Voyage AI client.
- `build_bm25_index(texts)`: Builds a BM25 index from a list of texts.

### retrieve.py
- `embed_query(query)`: Embeds a query using the Voyage AI client.
- `embed_texts(texts)`: Embeds a list of texts using the Voyage AI client.
- `count_tokens_for_messages(messages, model="gemini-1.5-pro")`: Estimates the total number of tokens used by a list of messages.
- `generate_answer(query, chunks)`: Generates an answer to the query using the provided chunks as context.
- `retrieve_chunks(query, index, texts, bm25, k=20)`: Retrieves relevant chunks for a query using both embedding and BM25 methods.

### run_pipeline.py
- `process_pdf(pdf_path)`: Processes a PDF file by extracting text, chunking, and building an index.
- `embed_query(query)`: Embeds a query using the Voyage AI client.
- `embed_texts(texts)`: Embeds a list of texts using the Voyage AI client.
- `retrieve_chunks(query, index, texts, bm25, k=20)`: Retrieves relevant chunks for a query using both embedding and BM25 methods.
- `generate_answer(query, chunks)`: Generates an answer to the query using the provided chunks as context.
- `start_chat_interface()`: Starts a Tkinter-based chat interface for querying.
- `main(input_path)`: Main function to process PDFs and start the chat interface.

### config.py
- `Config`: Configuration class to manage API keys and other settings.
- `validate_keys()`: Validates that all necessary API keys are set.
