import sys
import os
import logging
import PyPDF2


def extract_text(pdf_path, output_file):
    """Extract text from a PDF file and save it to a specified output file."""
    logging.info(f"Extracting text from {pdf_path}...")
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''  # Ensure text is not None
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logging.info(f"Extracted text saved to {output_file}")
    except FileNotFoundError:
        logging.error(f"File not found: {pdf_path}")
    except Exception as e:
        logging.error(f"An error occurred while extracting text: {e}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python extract_text.py <pdf_path> <output_text_file>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_file = sys.argv[2]
    extract_text(pdf_path, output_file)
