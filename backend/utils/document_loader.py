import pandas as pd
import pdfplumber
import os

def load_document(file_path: str) -> str:
    """
    Loads text content from PDF, CSV, or TXT.
    Returns a single string containing the document text.
    """
    if not os.path.exists(file_path):
        raise ValueError("File not found: " + file_path)

    # PDF
    if file_path.lower().endswith(".pdf"):
        full_text = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for pg in pdf.pages:
                    text = pg.extract_text()
                    if text:
                        full_text.append(text)
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")

        return "\n\n".join(full_text).strip()

    # CSV
    if file_path.lower().endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

    # TXT
    if file_path.lower().endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    raise ValueError("Unsupported file type. Only PDF, CSV, TXT are allowed.")
