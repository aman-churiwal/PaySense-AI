"""Load and index the payroll knowledge base into the vector store."""

import os

from src.rag.vector_store import VectorStore

def _chunk_markdown(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split Markdown text into overlapping chunks by paragraphs

    Splits on double newlines (paragraph boundaries), then merges small paragraphs and splits large ones to stay near chunk_size.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # If a single paragraph exceeds chunk_size, split it by sentences
            if len(para) > chunk_size:
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > chunk_size:
                        chunks.append(current_chunk.strip())
                        # Keep overlap
                        overlap_words = current_chunk.strip().split()[-10:]
                        current_chunk = " ".join(overlap_words) + " " + word
                    else:
                        current_chunk = (current_chunk + " " + word).strip()
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def load_knowledge_base(vector_store: VectorStore, kb_dir: str) -> int:
    """
    Load all Markdown files from the knowledge base directory into the vector store.

    Args:
        vector_store: The VectorStore to index into.
        kb_dir: Path to the knowledge_base directory containing .md files.

    Returns:
        Total number of chunks indexed.
    """
    if not os.path.isdir(kb_dir):
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    total_chunks = 0

    for filename in sorted(os.listdir(kb_dir)):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(kb_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        topic = filename.replace(".md", "")
        chunks = _chunk_markdown(content)

        metadatas = [
            {"source": "knowledge_base", "topic": topic, "filename": filename}
            for _ in chunks
        ]

        vector_store.add_documents(
            doc_id=f"kb_{topic}",
            chunks=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    return total_chunks