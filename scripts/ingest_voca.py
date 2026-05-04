"""
Pinecone voca-index 데이터 삽입 스크립트

Usage:
    python scripts/ingest_voca.py <file1> [file2 ...]

지원 파일 형식: .txt, .pdf, .docx
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

PINECONE_INDEX_NAME = os.getenv("PINECONE_VOCA_INDEX", "voca-index")
EMBEDDING_MODEL = os.getenv("UPSTAGE_EMBEDDING_MODEL", "solar-embedding-1-large")
CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "200"))


def get_loader(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(file_path), encoding="utf-8")
    elif ext == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            print("PDF 로드에 pypdf 패키지가 필요합니다: pip install pypdf")
            sys.exit(1)
        return PyPDFLoader(str(file_path))
    elif ext == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError:
            print("DOCX 로드에 docx2txt 패키지가 필요합니다: pip install docx2txt")
            sys.exit(1)
        return Docx2txtLoader(str(file_path))
    else:
        print(f"지원하지 않는 파일 형식입니다: {ext} (지원: {', '.join(SUPPORTED_EXTENSIONS)})")
        sys.exit(1)


def ingest(file_paths: list[Path]):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_upstage import UpstageEmbeddings
    from langchain_pinecone import PineconeVectorStore

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_docs = []
    for file_path in file_paths:
        print(f"로드 중: {file_path}")
        loader = get_loader(file_path)
        docs = loader.load_and_split(text_splitter=text_splitter)
        print(f"  → {len(docs)}개 청크 생성")
        all_docs.extend(docs)

    if not all_docs:
        print("삽입할 문서가 없습니다.")
        sys.exit(1)

    print(f"\n총 {len(all_docs)}개 청크를 Pinecone '{PINECONE_INDEX_NAME}'에 삽입 중...")

    embedding = UpstageEmbeddings(model=EMBEDDING_MODEL)
    PineconeVectorStore.from_documents(
        documents=all_docs,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME,
    )

    print("완료!")


def main():
    parser = argparse.ArgumentParser(description="voca-index에 문서를 삽입합니다.")
    parser.add_argument(
        "files",
        nargs="+",
        help=f"삽입할 파일 경로 (지원 형식: {', '.join(SUPPORTED_EXTENSIONS)})",
    )
    args = parser.parse_args()

    file_paths = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"파일을 찾을 수 없습니다: {f}")
            sys.exit(1)
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"지원하지 않는 파일 형식: {f} (지원: {', '.join(SUPPORTED_EXTENSIONS)})")
            sys.exit(1)
        file_paths.append(p)

    ingest(file_paths)


if __name__ == "__main__":
    main()
