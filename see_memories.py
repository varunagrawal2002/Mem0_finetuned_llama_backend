from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_multiuser_db")  # Or use host/port for remote

# Get points info
response = client.scroll(
    collection_name="mem0_multiuser_llama",
    limit=10,       # number of points to fetch
    with_payload=True
)

print(response)