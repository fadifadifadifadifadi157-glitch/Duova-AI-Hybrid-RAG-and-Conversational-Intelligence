import os
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def index_pdf(pdf_path, progress_callback=None, embedding_model=None):
    """
    Indexes a PDF file into ChromaDB with fine-grained progress reporting
    and automatic deduplication of existing document chunks.
    
    :param pdf_path: Path to the PDF file
    :param progress_callback: Optional callable(percent: int, message: str)
    :param embedding_model: Optional pre-loaded HuggingFaceEmbeddings instance
    """
    filename = os.path.basename(pdf_path)

    # Step 1: Read PDF
    if progress_callback:
        progress_callback(5, f"Reading '{filename}'...")
    time.sleep(0.05)

    data = PyPDFLoader(pdf_path)
    docs = data.load()

    if progress_callback:
        progress_callback(25, f"Extracted {len(docs)} pages from document...")
    time.sleep(0.05)

    # Step 2: Split into chunks
    if progress_callback:
        progress_callback(40, "Splitting into semantic chunks...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Tag every chunk with clean filename and page info
    for chunk in chunks:
        chunk.metadata["source_name"] = filename

    total_chunks = len(chunks)
    if progress_callback:
        progress_callback(55, f"Generated {total_chunks} chunks. Preparing ChromaDB...")
    time.sleep(0.05)

    # Step 3: Embeddings & Vector store
    if embedding_model is None:
        if progress_callback:
            progress_callback(60, "Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vectorstore = Chroma(
        persist_directory="ChromaDB",
        embedding_function=embedding_model
    )

    # Multi-document mode: only remove existing chunks for THIS file (deduplication).
    # Other PDFs already in ChromaDB are left completely untouched.
    try:
        existing = vectorstore.get(where={"source_name": filename})
        if existing and existing.get("ids"):
            vectorstore.delete(ids=existing["ids"])
            if progress_callback:
                progress_callback(62, f"Removed {len(existing['ids'])} old chunks for '{filename}'...")
    except Exception:
        pass

    # Step 4: Batch add documents with animated progress from 65% to 95%
    batch_size = 50
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        if progress_callback:
            current_done = min(i + len(batch), total_chunks)
            pct = 65 + int(30 * (current_done / max(total_chunks, 1)))
            progress_callback(
                min(pct, 95),
                f"Computing embeddings & storing: {current_done}/{total_chunks} chunks ({pct}%)..."
            )
            time.sleep(0.02)

    # Step 5: Completed
    if progress_callback:
        progress_callback(100, f"Successfully indexed '{filename}' ({total_chunks} chunks ready)!")
    time.sleep(0.1)

    print(f"PDF '{filename}' indexed successfully ({total_chunks} chunks)!")
    return {"filename": filename, "num_pages": len(docs), "num_chunks": total_chunks}