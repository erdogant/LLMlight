from qdrant_client import QdrantClient

# Initialize the client
client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

# Prepare your documents, metadata, and IDs
docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations", "Another sentence"]
metadata = [
    {"source": "Langchain-docs"},
    {"source": "Linkedin-docs"},
    {"source": "personal docs"},
]
ids = [42, 2, 1]

# Use the new add method
client.add(
    collection_name="demo_collection",
    documents=docs,
    metadata=metadata,
    ids=ids
)


#%%
search_result = client.query(collection_name="demo_collection", query_text="another")
print(search_result)
