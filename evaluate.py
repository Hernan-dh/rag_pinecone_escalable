"""Calcula Precision@5 y Recall@5 para el Golden Set local."""
import json
from pathlib import Path

from rag_system import RAGSystem

K = 5


def main() -> None:
    golden_set = json.loads(Path("golden_set.json").read_text(encoding="utf-8"))
    rag = RAGSystem(k=K)
    precisions, recalls = [], []

    for item in golden_set:
        docs = rag.search(item["pregunta"])
        retrieved_ids = [doc.metadata["source_id"] for doc in docs]
        relevant = item["documento_id_esperado"]
        hits = sum(source_id == relevant for source_id in retrieved_ids)
        precision = hits / K
        recall = 1.0 if hits else 0.0
        precisions.append(precision)
        recalls.append(recall)
        print(f"- {item['pregunta']}\n  recuperados={retrieved_ids} | P@{K}={precision:.2f} R@{K}={recall:.2f}")

    print("\nResumen de evaluación")
    print(f"Precision@{K}: {sum(precisions) / len(precisions):.2f}")
    print(f"Recall@{K}:    {sum(recalls) / len(recalls):.2f}")


if __name__ == "__main__":
    main()

