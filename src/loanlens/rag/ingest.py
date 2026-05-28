"""
CFPB regulatory PDF ingestion pipeline.
Loads PDFs, chunks them, embeds with sentence-transformers,
and stores in ChromaDB for retrieval.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from loguru import logger
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


def get_embedding_model():
    """Load sentence-transformers embedding model. Runs on CPU, free."""
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.embedding_model}")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def get_chroma_client():
    """Get persistent ChromaDB client."""
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
    )
    return client


def get_or_create_collection(client):
    """Get or create the CFPB regulations collection."""
    settings = get_settings()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def load_and_chunk_pdfs(pdf_directory: str) -> list[dict]:
    """
    Load all PDFs from directory and split into chunks.

    Args:
        pdf_directory: Path to folder containing CFPB PDFs

    Returns:
        List of chunk dicts with text, metadata
    """
    settings = get_settings()
    pdf_dir = Path(pdf_directory)
    pdf_paths = list(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {pdf_directory}")

    logger.info(f"Found {len(pdf_paths)} PDFs: {[p.name for p in pdf_paths]}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    all_chunks = []

    for pdf_path in pdf_paths:
        logger.info(f"Loading {pdf_path.name}...")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            logger.info(f"  Loaded {len(pages)} pages")

            # Split into chunks
            chunks = splitter.split_documents(pages)
            logger.info(f"  Split into {len(chunks)} chunks")

            for chunk in chunks:
                # Clean up text
                text = chunk.page_content.strip()
                if len(text) < 50:  # Skip very short chunks
                    continue

                all_chunks.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": chunk.metadata.get("page", 0),
                })

        except Exception as e:
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            continue

    logger.info(f"Total chunks after filtering: {len(all_chunks)}")
    return all_chunks


def generate_chunk_id(text: str, source: str, page: int, index: int = 0) -> str:
    """Generate stable unique ID for a chunk."""
    content = f"{source}:{page}:{index}:{text[:200]}"
    return hashlib.md5(content.encode()).hexdigest()


def ingest_to_chromadb(
    pdf_directory: str,
    force_reingest: bool = False
) -> int:
    """
    Full ingestion pipeline: PDF -> chunks -> embeddings -> ChromaDB.

    Args:
        pdf_directory: Path to CFPB PDFs
        force_reingest: If True, delete and reingest all chunks

    Returns:
        Number of chunks indexed
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check if already ingested
    existing_count = collection.count()
    if existing_count > 0 and not force_reingest:
        logger.info(
            f"ChromaDB already has {existing_count} chunks. "
            f"Use force_reingest=True to reingest."
        )
        return existing_count

    if force_reingest and existing_count > 0:
        settings = get_settings()
        client.delete_collection(settings.chroma_collection_name)
        collection = get_or_create_collection(client)
        logger.info("Deleted existing collection for reingestion")

    # Load and chunk PDFs
    chunks = load_and_chunk_pdfs(pdf_directory)

    # Get embeddings
    embedding_model = get_embedding_model()

    # Batch insert into ChromaDB
    batch_size = 100
    total_inserted = 0

    logger.info(f"Inserting {len(chunks)} chunks into ChromaDB...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [c["text"] for c in batch]
        ids = [
            generate_chunk_id(c["text"], c["source"], c["page"], i + idx)
            for idx, c in enumerate(batch)
        ]
        metadatas = [
            {"source": c["source"], "page": c["page"]}
            for c in batch
        ]

        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)

        # Insert into ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_inserted += len(batch)
        logger.info(f"  Inserted {total_inserted}/{len(chunks)} chunks...")

    logger.info(f"Ingestion complete — {total_inserted} chunks in ChromaDB")
    return total_inserted


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    settings = get_settings()
    count = ingest_to_chromadb(
        pdf_directory="data/regulations",
        force_reingest=False
    )
    print(f"\nDone — {count} regulatory chunks indexed")
