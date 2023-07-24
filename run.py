import os
import json
import streamlit as st
import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Qdrant

from dotenv import load_dotenv
load_dotenv(".env")


st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.header("Welcome To Home Law")

# connect to a Qdrant Cluster
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# create a new collection(DB)
vectors_config = qdrant_client.http.models.VectorParams(
    size=1536,
    distance=qdrant_client.http.models.Distance.COSINE   
)

client.recreate_collection(
    collection_name = os.getenv("QDRANT_COLLECTION_NAME"),
    vectors_config=vectors_config
)


embeddings = OpenAIEmbeddings()

# Connect to vector collection store
vector_store = Qdrant(
    client=client,
    collection_name = os.getenv("QDRANT_COLLECTION_NAME"),
    embeddings=embeddings
)


# with open("embbed_data.json", encoding="utf8") as f:
#     data = json.load(f)
#     
data = ["Ball", "elephant", "color red", "Ronaldo", "America"]

BATCH_SIZE = 1
print(f"Number of documents: {len(data)}")
doc_chunks_list = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

for i in range(0, len(doc_chunks_list)):
    print(f"Loading batch number {i + 1}...")

    Qdrant.add_texts( # we call the obj direct not vector_store.add_texts
        self=vector_store,
        texts=doc_chunks_list[i],
              
    )
print("Finished loading documents to Qdrant")



query = "Government of Nigeria"
docs = vector_store.similarity_search(query)
print("Similarity Search: ",docs[:3])

