import os
import streamlit as st
import qdrant_client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

from dotenv import load_dotenv
load_dotenv(".env")


# Streamlit Component
st.set_page_config(
    page_title="USA Law Code",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.header("USA Laws, Codes")
user_city = st.selectbox("Select a City", ("Maricopa", "LAH"))
user_chat = st.text_input("You: ", key=input)
submit = st.button("Browse Law Code")


# connect to a Qdrant Cluster
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)


embeddings = OpenAIEmbeddings()


def similarity_search():
    try:
        db = user_city
        user_input = user_chat

        if submit:
            if db == "LAH":
                db = "collection_two"  # I.e set a collection/DB name
                print("Yes")
            elif db == "Maricopa":
                db = "Maricopa"

            # Connect to vector collection store
            print("Db:", db)
            vector_store = Qdrant(
                client=client,
                collection_name=db,
                embeddings=embeddings
            )
            docs = vector_store.similarity_search(user_input)
            st.subheader("Top Matches:")
            st.text(docs[0].page_content)
            st.text(docs[1].page_content)
    except Exception as e:
        st.error(e)


similarity_search()

# streamlit run app.py
