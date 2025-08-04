#### Building RAG chatbot with langchain and gemini pro (Getting idea of how to build the interface with Gemini 1.5 Pro)
#### https://www.youtube.com/watch?v=WeVb0cE3rrs

#### Storing everything in ChromaDB
#### https://medium.com/keeping-up-with-ai/how-i-built-a-rag-based-ai-chatbot-from-my-personal-data-88eec0d3483c

from dotenv import load_dotenv
from langchain.schema import Document

import pandas as pd
import numpy as np

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import torch

#### For interactive RAG chatbot
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from typing import Callable
from pydantic import BaseModel

CHROMA_PATH = "chroma"
COLLECTION_PATH = "langchain"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", DEVICE)

class MyEmbeddingFunction:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input):
        # Chroma may call this if it supports function-style
        return self.embedding_model.encode(input, convert_to_tensor=False).tolist()

    def embed_query(self, input):
        return self.embedding_model.encode(input, convert_to_tensor=False).tolist()
    
    def embed_documents(self, input):
        return self.embedding_model.encode(input, convert_to_tensor=False).tolist()
        
def creating_chromadb(alldata):
    print("Generating the Chroma vector DB....")

    embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    
    embedding_function = MyEmbeddingFunction(embeddings)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    client.delete_collection(name=COLLECTION_PATH)
    collection = client.create_collection(name=COLLECTION_PATH, embedding_function=embedding_function)

    documents = []
    ### Save all reports
    for i, row in alldata.iterrows():
        content = (
            f"MDR_REPORT_KEY: {row['MDR_REPORT_KEY']}\n"
            f"YEAR: {int(row['YEAR'])}\n"
            f"PATIENT_SEX: {row['PATIENT_SEX']}\n"
            f"BRAND_NAME: {row['BRAND_NAME']}\n"
            f"ADVERSE_EVENTS: {str(row["ADVERSE_EVENTS"])}\n"
        )

        metadata = {
            "MDR_REPORT_KEY": str(row['MDR_REPORT_KEY']),
            "YEAR": str(row['YEAR']),
            "PATIENT_SEX": str(row['PATIENT_SEX'])
        }
        documents.append(Document(page_content=content, metadata=metadata))

    ### Trends analysis (for patient sex only)
    trend_df = alldata.groupby(["YEAR", "PATIENT_SEX"]).size().reset_index(name="count")
    for sex in trend_df["PATIENT_SEX"].unique():
        group = trend_df[trend_df["PATIENT_SEX"] == sex]
        trend_summary = "\n".join([
            f"In {row['YEAR']}, there were {row['count']} reports for sex {sex}."
            for _, row in group.iterrows()
        ])

        trend_doc = (
            f"Trend Summary:\n"
            f"Patient Sex: {sex}\n"
            f"Adverse event reports per year:\n{trend_summary}"
        )

        metadata = {
            "MDR_REPORT_KEY": f"adverse_events_by_year_{sex}_{row['YEAR']}", # just a fake one
            "type": "trend_summary",
            "PATIENT_SEX": sex
        }
        documents.append(Document(page_content=trend_doc, metadata=metadata))


    chroma_documents_content = [doc.page_content for doc in documents]
    chroma_metadatas = [doc.metadata for doc in documents]
    chroma_ids = [str(doc.metadata["MDR_REPORT_KEY"]) for doc in documents]
    collection.upsert(
        documents=chroma_documents_content,
        metadatas=chroma_metadatas,
        ids=chroma_ids
    )

    # batches = create_batches(
    #     api=client,
    #     ids=chroma_ids,
    #     documents=chroma_documents_content,
    #     metadatas=chroma_metadatas,
    #     embeddings=None 
    # )

    # batch_count = 0
    # for batch_ids, _, batch_metadatas, batch_documents in batches:
    #     batch_count += 1
    #     print(f"Processing batch {batch_count} of size {len(batch_documents)}...")
    #     collection.upsert(
    #         documents=batch_documents,
    #         metadatas=batch_metadatas,
    #         ids=batch_ids
    #     )
    #     print(f"Batch {batch_count} upserted.")
    print("Done....")

class CustomRetriever(BaseRetriever):
    vectorstore: Chroma
    filter_func: Callable  = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Optional: Extract a doc_id from query
        doc_id = None
        match = re.search(r"\b\d{8}\b", query)
        if match:
            doc_id = match.group()

        # Filter metadata if applicable
        search_kwargs = {"k": 1}
        if doc_id:
            ### Metadata filtering
            results = collection.get(
                where={"MDR_REPORT_KEY": doc_id},
                # where={"$and": [
                #     {"MDR_REPORT_KEY": doc_id},
                #     # {"PATIENT_SEX": "Male"}, 
                #     # {"YEAR": "2022"}       
                # ]}, 
                include=["documents", "metadatas"]
            )

            docs = [
                Document(page_content=doc, metadata=metadata)
                for doc, metadata in zip(results["documents"], results["metadatas"])
            ]
            print(f"doc_id {doc_id} found")
            return docs

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        return retriever.get_relevant_documents(query)


##### Accuracy Test: Simply grab lists of the report key and test the gender it retrieved from the database
def retriever(report_key=0):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_PATH)
    embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    embedding_function = MyEmbeddingFunction(embeddings)
    ### 
    user_query = f"What is the gender for patient, {report_key}"
    query_embedding = embedding_function.embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    ### Metadata filtering
    results = collection.get(
        where={"$and": [
            {"MDR_REPORT_KEY": "13926733"},
            # {"PATIENT_SEX": "Male"}, 
            # {"YEAR": "2022"}       
        ]}, 
        include=["documents", "metadatas"]
    )
    ### collection.get(ids=[str(x)])
    return results
    

load_dotenv()

alldata = pd.read_excel('for google cloud.xlsx', sheet_name="Sheet1")
# print("Creating Chroma VectorDB....")
# creating_chromadb(alldata)
# print("Done....")

custom_retriever = None
print("Getting the collection....")
tqdm.pandas(desc="my bar!")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_PATH)
docs = collection.get() 
# print(type(docs), "docs", docs) 
print(f"\nTotal number of documents: {collection.count()}")

#############################################################
#### Building the interactive RAG chatbot with Gemini 1.50-pro
print("Building RAG chatbot.....")

### set a title for chatbot only
st.title("RAG chatbot for coronary drug-eluting stent from MAUDE")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

data = None  
# Check if data is loaded before processing
if data is None:
    embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") # , device=DEVICE
    embedding_function = MyEmbeddingFunction(embeddings)
    
    vectorstore = Chroma(
        collection_name=COLLECTION_PATH,  
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH        
    )

    # Input box
    query = st.chat_input("Ask me something...")
    custom_retriever = CustomRetriever(vectorstore=vectorstore)
        
    # Set up the retriever with similarity search
    st.session_state.retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    st.success("Done")

# Show previous chat messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if query:  # Ensure retriever is defined
    # Show user message instantly
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Display "thinking" message from assistant
    with st.chat_message("assistant"):
        response_placeholder = st.empty() # st.spinner()
        response_placeholder.markdown("_Thinking..._") 

        # === Build the full RAG chain here ===
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro", 
                temperature=0.1,
                max_output_tokens=300)
        
        
        # Set up the question-answer chain
        question_answer_chain = create_stuff_documents_chain(
            llm, 
            prompt

        )
        # rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)
        custom_retriever = CustomRetriever(vectorstore=vectorstore)
        rag_chain = create_retrieval_chain(custom_retriever, question_answer_chain)

        # Get the response from the RAG chain
        response = rag_chain.invoke({"input": query})

        # Update placeholder with real response
        response_placeholder.markdown(response['answer'])
        # Store the assistant's response in history
        st.session_state.history.append({"role": "assistant", "content": response['answer']})

        # Save user query and assistant response to a text file
        with open("chat_history.txt", "a", encoding="utf-8") as f:
            f.write("User: " + query + "\n")
            f.write("Assistant: " + response["answer"] + "\n\n")


### Run streamlit app
### python -m streamlit run rag-final.py


### Just some example questions that you can ask Gemini Chatbot
# 1. What were the adverse events reported in MDR Report Key 20211227? 
# 2. What was the patient's sex in MDR Report Key 18141819? 
# 3. In what year was MDR Report Key 18029867 reported? 
# 4. What types of adverse events were reported across all MDR reports? 
# Can you list all the adverse events were reported across all MDR reports?
# 5. How many MDR reports are from 2024?
# Give me random 5 reports about thrombosis. List in format, (id, gender, year)




