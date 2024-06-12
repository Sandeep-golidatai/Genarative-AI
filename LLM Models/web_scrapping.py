import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI,  GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv('GOOGLE_API_KEY')

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

# Check if the API key is available
if not api_key:
    st.error("Google API key is missing. Please set the GOOGLE_API_KEY environment variable.")
else:
    if process_url_clicked:
        try:
            # Load data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data loading... started... ✅✅✅")
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text splitting... started... ✅✅✅")
            docs = text_splitter.split_documents(data)

            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="text-embed-gig-base")
            vectorstores = FAISS.from_documents(docs, embeddings)

            main_placeholder.text("Vector embedding... started... ✅✅✅")

            # Save FAISS index in a pickle file as a database
            file_path = "vectorstores_genai.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(vectorstores, f)

            main_placeholder.text("Processing complete... ✅✅✅")

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                vectorstores = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(
                    # llm=ChatGoogleGenerativeAI(api_key=api_key, model="chat-bison-001"),
                    llm=GoogleGenerativeModel(api_key=api_key, model="gemini-pro"),
                    retriever=vectorstores.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)
                st.header("Answer")
                st.subheader(result["answer"])



 