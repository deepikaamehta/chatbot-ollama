from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import json
import uuid

# Connect to your Qdrant container
client = QdrantClient(host="localhost", port=6333)

# Define collection name and vector size (use length of first embedding)
collection_name = "uploads"

with open("embedded_chunks.json", "r") as f:
    data = json.load(f)

# Create collection if not exists
if not client.collection_exists(collection_name):
    vector_size = len(data[0]["embedding"])
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine")
    )

# Insert points into collection
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=item["embedding"],
        payload={"text": item["text"], "file_name": item["file_name"]}
    )
    for item in data
]

client.upsert(collection_name=collection_name, points=points)

print(f"✅ Uploaded {len(points)} vectors to Qdrant collection '{collection_name}'")



