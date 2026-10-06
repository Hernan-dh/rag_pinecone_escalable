"""Crea el índice serverless compatible con text-embedding-3-small."""
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

DIMENSION = 1536


def main() -> None:
    load_dotenv()
    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ["INDEX_NAME"]
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
        print(f"Índice '{index_name}' creado (serverless, dimensión {DIMENSION}).")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    else:
        description = pc.describe_index(index_name)
        if description.dimension != DIMENSION:
            raise ValueError(f"El índice tiene dimensión {description.dimension}; se requieren {DIMENSION}.")
        print(f"El índice '{index_name}' ya existe y es compatible.")


if __name__ == "__main__":
    main()

