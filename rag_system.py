"""Recuperador híbrido: Pinecone semántico + BM25 léxico."""
import os

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from ingest import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGSystem:
    def __init__(self, k: int = 5) -> None:
        load_dotenv()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2400, chunk_overlap=300)
        local_chunks = splitter.split_documents(load_documents())

        bm25 = BM25Retriever.from_documents(local_chunks)
        bm25.k = k
        vector_store = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"],
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            namespace=os.getenv("PINECONE_NAMESPACE", "documentacion-tecnica"),
        )
        semantic = vector_store.as_retriever(search_kwargs={"k": k})
        self.retriever = EnsembleRetriever(
            retrievers=[semantic, bm25], weights=[0.6, 0.4]
        )
        self.k = k

    def search(self, query: str):
        """Devuelve hasta k documentos fusionando ambos rankings."""
        return self.retriever.invoke(query)[: self.k]

