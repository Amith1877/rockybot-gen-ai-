import os
import streamlit as st
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GOOGLE_API_KEY = "AIzaSyAX9byEmAvQvhgokEYNZaa6Hh9tPxFL0Zc"

st.set_page_config(page_title="RockyBot", page_icon="📈")
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
folder_path = "faiss_store_gemini"
main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

if process_url_clicked:
    urls = [url for url in urls if url]

    if not urls:
        st.sidebar.error("Please enter at least one URL!")
    else:
        try:
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)

            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            vectorstore = FAISS.from_documents(docs, embeddings)
            time.sleep(2)

            # ✅ Save using FAISS built-in method instead of pickle
            vectorstore.save_local(folder_path)

            main_placeholder.text("Processing Complete! You can now ask questions ✅")

        except Exception as e:
            st.error(f"Error processing URLs: {e}")

query = st.text_input("Ask a Question:")

if query:
    if not os.path.exists(folder_path):
        st.error("Please process URLs first!")
    else:
        # ✅ Load using FAISS built-in method instead of pickle
        vectorstore = FAISS.load_local(
            folder_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the context below.
        Be detailed and accurate.
        At the end, mention the sources used.

        Context: {context}

        Question: {question}

        Answer:
        """)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Fetching answer..."):
            result = chain.invoke(query)

        st.header("Answer")
        st.write(result)