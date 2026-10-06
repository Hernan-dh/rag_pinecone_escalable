"""Procesa Markdown, crea chunks y los inserta en Pinecone."""
from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")


def load_documents() -> list[Document]:
    """Cada archivo Markdown es una fuente; page=1 aplica a este dataset no paginado."""
    documents = []
    for path in sorted(DATA_DIR.glob("*.md")):
        front_matter, body = path.read_text(encoding="utf-8").split("---\n", 2)[1:]
        metadata = {"source_id": path.stem, "source": path.name, "page": 1}
        for line in front_matter.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
        documents.append(Document(page_content=body.strip(), metadata=metadata))
    return documents


def main() -> None:
    load_dotenv()
    index_name = os.environ["INDEX_NAME"]
    namespace = os.getenv("PINECONE_NAMESPACE", "documentacion-tecnica")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2400, chunk_overlap=300)
    chunks = splitter.split_documents(load_documents())
    for number, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source_id']}-{number}"
        # Además del campo `text` que maneja PineconeVectorStore, queda explícito
        # el contenido solicitado para consultas o inspección directa de metadatos.
        chunk.metadata["content"] = chunk.page_content

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    store.add_documents(chunks, ids=ids)
    print(f"{len(chunks)} chunks insertados en '{index_name}' / namespace '{namespace}'.")


if __name__ == "__main__":
    main()
