from __future__ import annotations

import time

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery


COLLECTION_NAME = "TextDocument"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def wait_for_weaviate(max_wait_seconds: int = 30) -> None:
    client = weaviate.connect_to_local()
    deadline = time.time() + max_wait_seconds
    try:
        while time.time() < deadline:
            if client.is_ready():
                return
            time.sleep(1)
    finally:
        client.close()
    raise RuntimeError("Weaviate is not ready. Check docker logs.")


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(COLLECTION_NAME):
        return

    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
        ],
    )


def main() -> None:
    wait_for_weaviate()
    model = SentenceTransformer(MODEL_NAME)

    texts = [
        "Kubernetes manages containerized workloads.",
        "Weaviate stores vectors for semantic search.",
        "Sentence Transformers create embeddings from text.",
        "Minikube is useful for local Kubernetes testing.",
    ]
    sources = ["k8s", "vectordb", "nlp", "local-dev"]

    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    query_vector = model.encode(
        "How do I run semantic vector search locally?",
        normalize_embeddings=True,
    ).tolist()

    client = weaviate.connect_to_local()
    try:
        ensure_collection(client)
        collection = client.collections.get(COLLECTION_NAME)

        for text, source, vector in zip(texts, sources, vectors, strict=True):
            collection.data.insert(
                properties={"text": text, "source": source},
                vector=vector,
            )

        result = collection.query.near_vector(
            near_vector=query_vector,
            limit=3,
            return_properties=["text", "source"],
            return_metadata=MetadataQuery(distance=True),
        )

        print("Inserted documents:", len(texts))
        print("Top-3 semantic search results")
        for idx, obj in enumerate(result.objects, start=1):
            text = obj.properties.get("text", "")
            source = obj.properties.get("source", "")
            distance = obj.metadata.distance if obj.metadata else None
            print(f"{idx}. distance={distance:.4f} source={source} text={text}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
