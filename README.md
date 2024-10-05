# Contextual Retriever

The Contextual Retriever is designed to process and retrieve information from multiple PDF files efficiently. It extracts text, chunks and contextualizes the content, builds an index, and allows for interactive querying.

## Installation

1. **Clone the Repository**: Clone this repository to your local machine.

2. **Install Dependencies**: Ensure you have Python installed. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Pipeline

To process a directory of PDF files, use the `run_pipeline.py` script. This script will extract text from each PDF, chunk and contextualize the text, build an index, and allow for interactive querying.

```bash
python run_pipeline.py <path_to_directory>
```

Replace `<path_to_directory>` with the path to the directory containing your PDF files.

### Data Flow and Outputs

1. **extract_text.py**: 
   - **Role**: Extracts text from PDF files.
   - **Output**: `document_text.txt` in the `output` directory.
   - **Next Step**: The extracted text is used by `chunk_text.py`.

2. **chunk_text.py**:
   - **Role**: Divides the extracted text into manageable chunks.
   - **Output**: `chunks.json` in the `output` directory.
   - **Next Step**: The chunks are used by `chunk_and_contextualize.py`.

3. **chunk_and_contextualize.py**:
   - **Role**: Adds contextual information to the chunks.
   - **Output**: `contextualized_chunks.json` in the `output` directory.
   - **Next Step**: The contextualized chunks are used by `build_index.py`.

4. **build_index.py**:
   - **Role**: Builds an index for efficient retrieval.
   - **Outputs**: `chunks.index` and `chunks_meta.json` in the `output` directory.
   - **Next Step**: The index and metadata are used by `retrieve.py`.

5. **retrieve.py**:
   - **Role**: Handles querying and retrieves relevant information.
   - **Inputs**: Uses `chunks.index` and `chunks_meta.json` to perform retrieval based on user queries.

### Contextual Retrieval Strategies

The system implements Contextual Retrieval strategies as outlined by Anthropic, enhancing traditional RAG methods with Contextual Embeddings and Contextual BM25:

- **Contextual Embeddings**: Each chunk of text is contextualized by prepending explanatory context before embedding. This context helps situate the chunk within the overall document, improving retrieval accuracy.

- **Contextual BM25**: A BM25 index is created using the contextualized chunks, allowing for precise lexical matching alongside semantic similarity searches.

These techniques reduce retrieval failures by providing both semantic and exact match capabilities, ensuring that the most relevant information is retrieved efficiently.

### Interactive Querying

After processing the PDFs, the system will be ready for queries. You can enter your queries interactively, and the system will retrieve relevant information based on the context.

## Notes

- Ensure all PDF files are placed in the specified directory before running the pipeline.
- The system is designed to handle large collections of documents efficiently.

For any issues or questions, please use Issues within this repository for support.