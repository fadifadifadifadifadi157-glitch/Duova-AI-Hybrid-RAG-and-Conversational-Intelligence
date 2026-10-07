"""
preindex.py — Default PDF Pre-Loader
--------------------------------------
Called automatically by duova_app.py at startup.
Checks whether the bundled default PDF(s) in 'preload_data/' are already
indexed in ChromaDB. If not, indexes them silently so employers/viewers
see a ready knowledge base the moment the app loads — no waiting required.
"""

import os
import sqlite3

# ── Path to the folder containing default PDFs ──
PRELOAD_DIR = os.path.join(os.path.dirname(__file__), "preload_data")
CHROMADB_PATH = os.path.join(os.path.dirname(__file__), "ChromaDB", "chroma.sqlite3")


def get_already_indexed_names() -> set:
    """Returns a set of filenames already indexed in ChromaDB."""
    try:
        conn = sqlite3.connect(CHROMADB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata WHERE key = 'source_name'"
        )
        names = {r[0] for r in cur.fetchall() if r[0]}
        conn.close()
        return names
    except Exception:
        return set()


def run_preindex(progress_callback=None):
    """
    Scans preload_data/ for PDFs and indexes any that aren't in ChromaDB yet.
    Returns list of filenames that were newly indexed.
    """
    if not os.path.isdir(PRELOAD_DIR):
        return []

    pdf_files = [
        f for f in os.listdir(PRELOAD_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        return []

    already_indexed = get_already_indexed_names()
    to_index = [f for f in pdf_files if f not in already_indexed]

    if not to_index:
        return []  # All default PDFs already in ChromaDB — nothing to do

    # Lazy import to avoid loading heavy models unless needed
    from langchain_huggingface import HuggingFaceEmbeddings
    from create_DB import index_pdf

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    newly_indexed = []
    for filename in to_index:
        pdf_path = os.path.join(PRELOAD_DIR, filename)
        try:
            if progress_callback:
                progress_callback(filename, 0, f"Starting '{filename}'...")
            index_pdf(
                pdf_path,
                progress_callback=(
                    (lambda pct, msg, fn=filename: progress_callback(fn, pct, msg))
                    if progress_callback else None
                ),
                embedding_model=embedding_model,
            )
            newly_indexed.append(filename)
            if progress_callback:
                progress_callback(filename, 100, f"'{filename}' ready!")
        except Exception as e:
            print(f"[preindex] Failed to index '{filename}': {e}")

    return newly_indexed
