import sys
import os
import logging
import PyPDF2
import docx  # Added import for handling .docx files

def extract_text(file_path, output_file):
    """Extract text from a PDF or DOCX file and save it to a specified output file."""
    logging.info(f"Extracting text from {file_path}...")
    try:
        ext = os.path.splitext(file_path)[1].lower()
        text = ''
        if ext == '.pdf':
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text() or ''
        elif ext == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + '\n'
        else:
            logging.error(f"Unsupported file type: {ext}")
            return

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct absolute path for the output file
        output_file_path = os.path.join(script_dir, output_file)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Extracted text saved to {output_file_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while extracting text: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_text.py <file_path> <output_text_file>")
        sys.exit(1)
    file_path = sys.argv[1]
    output_file = sys.argv[2]
    extract_text(file_path, output_file)
