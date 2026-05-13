import asyncio
import os
from pathlib import Path

from app.logger import get_logger

logger = get_logger("wespeak.ingest")

PINECONE_INDEX_NAME = os.getenv("PINECONE_VOCA_INDEX", "voca-index")
EMBEDDING_MODEL = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "200"))

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def _get_loader(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(file_path), encoding="utf-8")
    elif ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(str(file_path))
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def _validate_files(file_paths: list[str]) -> list[Path]:
    paths = []
    for f in file_paths:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {f}")
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"지원하지 않는 파일 형식: {f}")
        paths.append(p)
    return paths


async def ingest_voca(file_paths: list[str]) -> None:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_upstage import UpstageEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    try:
        paths = _validate_files(file_paths)
        logger.info("ingest start - files=%s", file_paths)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        all_docs = []
        for path in paths:
            logger.info("loading - file=%s", path)
            loader = _get_loader(path)
            docs = await asyncio.to_thread(loader.load_and_split, text_splitter=text_splitter)
            logger.info("chunks created - file=%s count=%d", path, len(docs))
            all_docs.extend(docs)

        if not all_docs:
            logger.warning("ingest aborted - no documents loaded")
            return

        logger.info("ingesting - total_chunks=%d index=%s", len(all_docs), PINECONE_INDEX_NAME)

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        existing_names = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME in existing_names:
            await asyncio.to_thread(lambda: pc.Index(PINECONE_INDEX_NAME).delete(delete_all=True))
            logger.info("existing data deleted - index=%s", PINECONE_INDEX_NAME)
        else:
            await asyncio.to_thread(pc.create_index,
                name=PINECONE_INDEX_NAME,
                dimension=4096,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("index created - index=%s", PINECONE_INDEX_NAME)

        embedding = UpstageEmbeddings(model=EMBEDDING_MODEL)
        await asyncio.to_thread(
            PineconeVectorStore.from_documents,
            documents=all_docs,
            embedding=embedding,
            index_name=PINECONE_INDEX_NAME,
        )

        logger.info("ingest completed - total_chunks=%d", len(all_docs))

    except Exception as e:
        logger.error("ingest failed - files=%s error=%s", file_paths, e, exc_info=True)
        raise

