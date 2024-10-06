# Contextual Retriever

This project implements a contextual retriever using Voyage AI and OpenAI embeddings for document retrieval and reranking. Inspired by Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) article, we rerank using the Voyage AI model and generate answers using the OpenAI API.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [File Descriptions](#file-descriptions)
  - [config.py](#configpy)
  - [chunk_and_contextualize.py](#chunk_and_contextualizepy)
  - [build_index.py](#build_indexpy)
  - [retrieve.py](#retrievepy)
  - [run_pipeline.py](#run_pipelinepy)
- [Usage](#usage)
  - [Running the Pipeline Step by Step](#running-the-pipeline-step-by-step)
  - [Running the Entire Pipeline](#running-the-entire-pipeline)
- [Customization](#customization)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

The Contextual Retriever is a pipeline designed to enhance retrieval accuracy by contextualizing document chunks and leveraging Voyage AI and OpenAI embeddings along with BM25 for retrieval and reranking. This approach is inspired by Anthropic's methodology on Contextual Retrieval.

## Prerequisites

- Python 3.7 or higher
- Voyage AI API Key
- OpenAI API Key

## Installation

1. **Clone the repository**:

   ```bash
   git clone [repository link]
   ```

2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes the following dependencies:
   - `openai`
   - `faiss`
   - `numpy`
   - `voyageai`
   - `rank_bm25`

3. **Set up your API keys in `config.py`**:

   Ensure that the environment variables `OPENAI_API_KEY` and `VOYAGE_API_KEY` are set. The application will raise an error if these are not configured.

   ```bash
   # Example of setting environment variables
   export OPENAI_API_KEY='your_openai_api_key'
   export VOYAGE_API_KEY='your_voyage_api_key'
   ```

## Pipeline Overview

The pipeline consists of the following steps:

1. **Text Extraction and Contextualization** (`chunk_and_contextualize.py`): 
   - **OpenAI API**: Used to generate contextual information for each text chunk. The `generate_context` function sends a prompt to the OpenAI API to create context for each chunk based on the overall document.
   
2. **Index Building** (`build_index.py`): 
   - **Voyage AI API**: Used to generate embeddings for each chunk. The `embed_text` function calls the Voyage AI API to obtain embeddings, which are then used to build a FAISS index for similarity search.
   - **BM25**: A lexical retrieval model used to build a BM25 index from the text chunks.

3. **Retrieval and Reranking** (`retrieve.py`): 
   - **Voyage AI API**: Used to generate embeddings for user queries. The `embed_query` function uses the Voyage AI API to create query embeddings.
   - **OpenAI API**: Used to generate comprehensive answers based on the top retrieved chunks. The `generate_answer` function sends the top chunks to the OpenAI API to produce a detailed response.

## File Descriptions

### config.py

- **Purpose**: Stores configuration variables such as API keys.
- **Usage**: Update this file with your Voyage AI and OpenAI API keys.
- **Outputs**: None.
- **Notes**: Includes error handling to ensure environment variables are set.

### chunk_and_contextualize.py

- **Execution Order**: **First**
- **Purpose**: Processes the input document to extract text, split it into chunks, and contextualize each chunk.
- **How it works**:
  - **Extract Text**: Reads the provided PDF document and extracts the textual content.
  - **Chunking**: Splits the extracted text into manageable chunks (e.g., 500 tokens each) for efficient processing.
  - **Contextualization**: Uses the OpenAI API to generate additional context for each chunk, improving retrieval accuracy.
- **Outputs**:
  - `output/contextualized_chunks.json`: A JSON file containing the list of contextualized chunks.
- **Relation to Next File**:
  - The output file `contextualized_chunks.json` is used by `build_index.py` to create the retrieval indexes.
- **Notes**: Now includes logging and error handling for file operations and API calls.

### build_index.py

- **Execution Order**: **Second**
- **Purpose**: Builds the retrieval indexes from the contextualized chunks.
- **How it works**:
  - **Load Chunks**: Reads `output/contextualized_chunks.json`.
  - **Embedding Generation**: Uses the Voyage AI API to generate embeddings for each chunk.
  - **FAISS Index Creation**: Builds a FAISS index for efficient similarity search using the generated embeddings.
  - **BM25 Index Creation**: Constructs a BM25 index for lexical retrieval.
- **Outputs**:
  - `output/chunks.index`: The FAISS index file.
  - `output/bm25_index.pkl`: Serialized BM25 index.
  - `output/chunks_meta.json`: Metadata containing the text of each chunk.
- **Relation to Next File**:
  - The generated indexes are used by `retrieve.py` for retrieval and reranking.
- **Notes**: Now includes logging and error handling for file operations.

### retrieve.py

- **Execution Order**: **Third**
- **Purpose**: Handles user queries by retrieving and reranking relevant chunks.
- **How it works**:
  - **Load Indexes**: Loads the FAISS index, BM25 index, and chunks metadata from the `output` directory.
  - **Embed Query**: Uses the Voyage AI API to generate an embedding for the user's query.
  - **Initial Retrieval**:
    - **FAISS Search**: Retrieves top chunks based on vector similarity.
    - **BM25 Search**: Retrieves top chunks based on lexical matching.
  - **Combine Results**: Merges and deduplicates results from both retrieval methods.
  - **Reranking**: Reranks the combined results using cosine similarity between the query embedding and chunk embeddings.
  - **Generate Answer**: Uses the OpenAI API to generate a comprehensive answer based on the top retrieved chunks.
  - **Output**: Presents the top relevant chunks and the generated answer as the response to the user's query.
- **Outputs**:
  - Displays the retrieved and reranked chunks and the generated answer related to the user's query.
- **Relation to Previous Files**:
  - Uses the indexes and metadata generated by `build_index.py`.

### run_pipeline.py

- **Execution Order**: **Wrapper Script (Optional)**
- **Purpose**: Runs the entire pipeline in sequence.
- **How it works**:
  - Calls `chunk_and_contextualize.py`, `build_index.py`, and `retrieve.py` in order.
- **Usage**:
  - Simplifies the process by allowing the user to execute one command to run the entire pipeline.
  - The input can be a single file or a directory containing multiple PDF files.

## Customization

To customize the project to use a different model, you will need to edit the following files:

1. **retrieve.py**:
   - Modify the `embed_query` function to use your desired model for generating query embeddings.
   - Update the `generate_answer` function if the model's API or response format differs.

2. **build_index.py**:
   - Change the `embed_text` function to use your chosen model for generating embeddings for document chunks.

3. **chunk_and_contextualize.py**:
   - Adjust the `generate_context` function to use a different model if needed, ensuring compatibility with the model's API.

Ensure that any changes made are consistent with the model's API and that all necessary dependencies are installed.

## Usage

### Running the Pipeline Step by Step

1. **Contextualize Chunks**:

   ```bash
   python chunk_and_contextualize.py /path/to/your/document.pdf
   ```

   - **Input**: PDF document.
   - **Output**: `output/contextualized_chunks.json`.

2. **Build Indexes**:

   ```bash
   python build_index.py
   ```

   - **Input**: `output/contextualized_chunks.json`.
   - **Outputs**:
     - `output/chunks.index`
     - `output/bm25_index.pkl`
     - `output/chunks_meta.json`

3. **Retrieve Information**:

   ```bash
   python retrieve.py "Your query here"
   ```

   - **Input**: User query.
   - **Outputs**: Relevant chunks and generated answer displayed in the console.

### Running the Entire Pipeline

Alternatively, you can use the `run_pipeline.py` script to execute all steps sequentially. The input can be a single file or a directory containing multiple PDF files:

```bash
python run_pipeline.py /path/to/your/input
```

- **Inputs**:
  - PDF document or a directory containing multiple PDF documents.
  - User query (entered during execution).
- **Outputs**:
  - All intermediate output files.
  - Retrieval results and generated answer displayed in the console.

## Acknowledgments

This project adopts many ideas from Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) article, with modifications to use Voyage AI's embeddings and reranking methods, and the OpenAI API for generating answers.