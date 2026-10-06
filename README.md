# RAG escalable con Pinecone

Módulo Python de recuperación híbrida: guarda chunks y metadatos en Pinecone Serverless, combina similitud vectorial con BM25 y evalúa el resultado con un Golden Set de cinco preguntas.

## Requisitos

Python 3.10+ y una cuenta de Pinecone y OpenAI. Se usa `text-embedding-3-small`, por lo que el índice requiere **1536 dimensiones**.

## Ejecución

1. Crear y activar un entorno virtual.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Completar `PINECONE_API_KEY` y `OPENAI_API_KEY` en `.env` (el archivo no se versiona; usar `.env.example` como referencia). Elegir `INDEX_NAME`, región y namespace.
4. Crear o verificar el índice Serverless: `python init_pinecone.py`.
5. Ingerir el dataset: `python ingest.py`.
6. Ejecutar la evaluación: `python evaluate.py`.

El script de inicialización crea el índice con métrica coseno, cloud/región configurables y dimensión 1536. La ingesta usa `RecursiveCharacterTextSplitter` (chunks de aproximadamente 600 tokens) y persiste en Pinecone el texto del chunk junto con `source_id`, `source`, `page`, `category`, `tags` y `chunk_id`. Todas las operaciones usan `PINECONE_NAMESPACE` para aislar el conjunto documental.

## Métricas

`evaluate.py` consulta el Golden Set y muestra por pregunta los documentos recuperados, además de:

- **Recall@5:** proporción de preguntas cuyo documento fuente esperado aparece entre los cinco resultados.
- **Precision@5:** proporción de los cinco resultados que pertenecen al documento fuente esperado.

Las métricas se calculan al ejecutar la evaluación porque dependen de los embeddings e índice creados con las credenciales de cada entorno.
