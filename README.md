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

### Customizing Settings

#### API Configuration

- **API Keys**: Set your OpenAI API key as an environment variable. This is done by adding the following line to your shell configuration file (e.g., `.bashrc`, `.zshrc`):
  ```bash
  export OPENAI_API_KEY='your-api-key-here'
  ```
  This ensures that your API key is securely accessed by the system without being hardcoded in the scripts.

### Customizing Model Choices

The Contextual Retriever allows you to customize the models used for embeddings and chat completions. By default, the system uses the following models:
- **Embeddings**: `text-embedding-3-large`
- **Chat Completions**: `gpt-4o-mini`

#### Changing the Models

To change the models used in the scripts, follow these steps:

1. **Open the Script**: Locate the script you wish to modify (`retrieve.py`, `chunk_and_contextualize.py`, or `build_index.py`).

2. **Modify the Model Name**: Find the line where the model is specified and replace it with your desired model. For example, to use a different embedding model, change:
   ```python
   model='text-embedding-3-large'
   ```
   to:
   ```python
   model='your-desired-model'
   ```

3. **Save the Changes**: After modifying the script, save your changes.

#### Available Models

Refer to the [OpenAI Models Documentation](https://platform.openai.com/docs/models) for a list of available models and their capabilities. Choose models that best fit your use case.

By customizing the models, you can optimize the performance and cost of the Contextual Retriever to suit your specific needs.

### Interactive Querying

After processing the PDFs, the system will be ready for queries. You can enter your queries interactively, and the system will retrieve relevant information based on the context.

## Notes

- Ensure all PDF files are placed in the specified directory before running the pipeline.
- The system is designed to handle large collections of documents efficiently.

For any issues or questions, please refer to the documentation or contact the support team.
