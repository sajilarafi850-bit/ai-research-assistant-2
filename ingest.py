"""
ingest.py

WHAT THIS FILE DOES (run this first, before app.py):
1. Reads every PDF inside the 'papers' folder
2. Splits each PDF into small chunks (~500 words each)
3. Converts each chunk into a vector (embedding)
4. Saves all vectors into a local database folder called 'chroma_db'

Run it like this in your terminal:
    python ingest.py

You only need to re-run this when you ADD NEW PDFs to the papers folder.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ----- SETTINGS -----
PAPERS_FOLDER = "papers"          # where your PDF files live
DB_FOLDER = "chroma_db"           # where the database will be saved
CHUNK_SIZE = 500                  # roughly how many characters per chunk
CHUNK_OVERLAP = 50                # slight overlap so we don't cut sentences awkwardly


def load_all_pdfs(folder_path):
    """
    STEP 1: Read every PDF in the folder.
    Each page of each PDF becomes one 'document' object with:
      - the text of that page
      - metadata: which file it came from, which page number
    This metadata is what lets us show citations later.
    """
    all_documents = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in '{folder_path}'")

    for filename in pdf_files:
        full_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(full_path)
        pages = loader.load()  # returns one entry per page, with page number attached
        print(f"  Loaded '{filename}' — {len(pages)} pages")
        all_documents.extend(pages)

    return all_documents


def split_into_chunks(documents):
    """
    STEP 2: Break long pages into smaller chunks.
    Why: a full page might be too long/unfocused for good retrieval.
    Smaller chunks = more precise matching to a question.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks total")
    return chunks


def build_vector_database(chunks):
    """
    STEP 3 + 4: Convert chunks into embeddings (numbers) and save them
    into a local Chroma database on disk.

    HuggingFaceEmbeddings runs locally on your machine — free, no API key needed.
    The specific model "all-MiniLM-L6-v2" is small, fast, and good enough for this project.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating embeddings and saving to database... (this may take a minute)")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_FOLDER,  # saves to disk so app.py can load it later
    )
    print(f"Done. Database saved to '{DB_FOLDER}'")
    return vector_db


if __name__ == "__main__":
    documents = load_all_pdfs(PAPERS_FOLDER)

    if len(documents) == 0:
        print(f"No PDFs found. Put your PDF files inside the '{PAPERS_FOLDER}' folder and re-run this script.")
    else:
        chunks = split_into_chunks(documents)
        build_vector_database(chunks)
        print("\nIngestion complete. You can now run: streamlit run app.py")