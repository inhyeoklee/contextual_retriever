import sys
import os
import PyPDF2

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""

        # Extract text from each page
        for page in reader.pages:
            text += page.extract_text()

    # Write the extracted text to 'document_text.txt'
    with open(os.path.join(output_dir, 'document_text.txt'), 'w') as f:
        f.write(text)