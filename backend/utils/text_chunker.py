from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(
    raw_text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
):
    """
    Splits raw text into chunks for embedding.
    
    Args:
        raw_text (str): The extracted document text.
        chunk_size (int): Maximum length of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        List[str]: A list of clean text chunks.
    """

    # ------------------------
    # Validate Input
    # ------------------------
    if not raw_text or len(raw_text.strip()) == 0:
        raise ValueError("Input text is empty. Cannot chunk an empty document.")

    # ------------------------
    # Configure Text Splitter
    # ------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            ".\n",
            " ",
            ""
        ]
    )

    # ------------------------
    # Split Text
    # ------------------------
    chunks = splitter.split_text(raw_text)

    # Clean up whitespace in each chunk
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return cleaned_chunks
