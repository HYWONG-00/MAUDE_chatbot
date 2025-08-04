#### Building RAG chatbot with langchain and gemini pro (Getting idea of how to build the interface with Gemini 1.5 Pro)
#### https://www.youtube.com/watch?v=WeVb0cE3rrs

#### Storing everything in ChromaDB
#### https://medium.com/keeping-up-with-ai/how-i-built-a-rag-based-ai-chatbot-from-my-personal-data-88eec0d3483c

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

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import Callable

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()


import os
from huggingface_hub import login
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HUGGINGFACE_TOKEN)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE", DEVICE)

TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
## for chroma vector database
CHROMA_PATH = "chroma"
COLLECTION_PRETRAINED_GISTPATH = "langchain-pretrained-GIST"

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

    embeddings = SentenceTransformer(TRANSFORMER_MODEL, device=DEVICE)
    
    embedding_function = MyEmbeddingFunction(embeddings)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    client.delete_collection(name=COLLECTION_PRETRAINED_GISTPATH)
    collection = client.create_collection(name=COLLECTION_PRETRAINED_GISTPATH, embedding_function=embedding_function)

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

    ### Trends analysis (for stent brands only)
    trend_df = alldata.groupby(["YEAR", "BRAND_NAME"]).size().reset_index(name="count")
    for brand in trend_df["BRAND_NAME"].unique():
        group = trend_df[trend_df["BRAND_NAME"] == brand]
        trend_summary = "\n".join([
            f"In {row['YEAR']}, there were {row['count']} reports for brand {brand}."
            for _, row in group.iterrows()
        ])

        trend_doc = (
            f"Trend Summary:\n"
            f"Stent brands: {brand}\n"
            f"Adverse event reports per year:\n{trend_summary}"
        )

        metadata = {
            "MDR_REPORT_KEY": f"adverse_events_by_year_{brand}_{row['YEAR']}", # just a fake one
            "type": "trend_summary",
            "BRAND_NAME": brand
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
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_PRETRAINED_GISTPATH)

        doc_id = None
        match = re.search(r"\b\d{8}\b", query)
        if match:
            doc_id = match.group()

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
            search_kwargs={"k": 1}
        )
        return retriever.get_relevant_documents(query)

### Metadata filtering
# results = collection.get(
#     where={"$and": [
#         {"MDR_REPORT_KEY": "13926733"},
#         # {"PATIENT_SEX": "Male"}, 
#         # {"YEAR": "2022"}       
#     ]}, 
#     include=["documents", "metadatas"]
# )
    
alldata = pd.read_excel('for rag.xlsx', sheet_name="Sheet1")
# print("Creating Chroma VectorDB....")
# creating_chromadb(alldata)
# print("Done....")

custom_retriever = None
print("Getting the collection....")
tqdm.pandas(desc="my bar!")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_PRETRAINED_GISTPATH)
docs = collection.get() 
# print(type(docs), "docs", docs) 
print(f"\nTotal number of documents: {collection.count()}")

#############################################################
#### Building the interactive RAG chatbot with fine-tuned GPT-2 LLM and allMiniLM sentence transformer
print("Building RAG chatbot.....")

### set a title for chatbot only
st.title("RAG chatbot for coronary drug-eluting stent adverse event analysis")

############ Done - Setting up fine-tuned GPT-2 model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model_id = "fine_tuned_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config= bnb_config,
                                             device_map="cpu") # {"" : 0}
model = get_peft_model(model, lora_config)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
############ Done - Loading fine-tuned GPT-2 model

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

data = None  
# Check if data is loaded before processing
if data is None:
    embeddings = SentenceTransformer(TRANSFORMER_MODEL) # , device=DEVICE
    embedding_function = MyEmbeddingFunction(embeddings)
    
    ### setup database, embed the whole original dataset with my embedding model
    vectorstore = Chroma(
        collection_name=COLLECTION_PRETRAINED_GISTPATH,  
        embedding_function=embedding_function,
        persist_directory=CHROMA_PATH        
    )

    query = st.chat_input("Ask me something...")
    ### set up the retriever - if you look into CustomRetriever, using SAME sentence transformer as the one for encoding dataset
    custom_retriever = CustomRetriever(vectorstore=vectorstore)

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
        print("User query:", query) 

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=custom_retriever
        )

        response = rag_chain.run(query)

        # Update placeholder with real response
        print("response", response)
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




