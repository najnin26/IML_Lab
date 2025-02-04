import PyPDF2
import pandas as pd

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    page_texts = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        page_texts.append((page_num + 1, text))
    return page_texts

# Function to convert extracted text to CSV
def text_to_csv(page_texts, csv_path):
    data = []
    for page_num, text in page_texts:
        lines = text.split('\n')
        for line in lines:
            if line.strip() != '':
                data.append([page_num, line])
    df = pd.DataFrame(data, columns=['Page', 'Text'])
    df.to_csv(csv_path, index=False)
    return df

# Paths to the PDF and CSV files
pdf_path = '02_Cloud-Intro.pdf'  # Update the path to your PDF file
csv_path = 'output.csv'

# Extract text from PDF and convert to CSV
page_texts = extract_text_from_pdf(pdf_path)
df = text_to_csv(page_texts, csv_path)

print(df.head())

print(f"PDF content has been successfully converted to CSV and saved at {csv_path}")