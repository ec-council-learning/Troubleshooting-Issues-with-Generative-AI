"""
This script creates a Streamlit web application that analyzes log files using natural language processing.
It leverages the LangChain library to process and analyze the content of uploaded log files.

Key features:
1. Allows users to upload a log file through a Streamlit interface.
2. Processes the log file using UnstructuredLoader to extract text.
3. Creates embeddings of the log content using OpenAIEmbeddings.
4. Stores the embeddings in a FAISS vector database for efficient retrieval.
5. Uses a ChatOpenAI model to analyze the log content based on a predefined prompt.
6. Presents the analysis results to the user through the Streamlit interface.
"""

# Import necessary libraries
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_unstructured import UnstructuredLoader
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Set up Streamlit page configuration
    st.set_page_config(page_title="Log Analyzer Bot")
    st.title("Log Analyzer Bot...💁 ")
    st.subheader("I can help you in Analyze Log data")
    
    # Create file uploader for log file
    logfile = st.file_uploader("Upload your log file", type=["txt"], accept_multiple_files=False)
    
    # Create analyze button
    submit = st.button("Analyze Log")
    
    if submit:
        with st.spinner('Wait for it...'):
            loganalyze(logfile)
        st.success("Hope I was able to save your time❤️")

def loganalyze(log_file):
    # Extract log file data
    print("Processing -", log_file.name)
    loader = UnstructuredLoader(log_file.name)
    pages = loader.load()
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(pages, embeddings)
    
    # Define prompt template
    #template = """You are analyzing a Linux Backup Log. The log contents are below:
    #template = """You are analyzing a Windows Diagnostic Log. The log contents are below:
    template = """You are analyzing a Mac Package Usage Log. The log contents are below:
    {context}
    Answer the following question from the contents of the Backup Log.
    {input}
    """
    prompt = PromptTemplate.from_template(template)
    
    # Set up language model
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create retrieval chain
    retriever = vector.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Invoke the chain with a specific question
    #response = retrieval_chain.invoke({"input": "Please detect if there are any issues in the backup and provide possible reasons"})
    #response = retrieval_chain.invoke({"input": "Please Provide a Summary of the Diagnostics status from the Log and if there are any issues, provide possible reasons and remediative commands"})
    response = retrieval_chain.invoke({"input": "Please Provide a Summary of the Diagnostics status from the Log and provide recommndations for the findings"})
    answer_content = response['answer']
    
    # Display the results
    st.write(answer_content)
    print("Here are the Analysis Results\n", answer_content)

if __name__ == '__main__':
    main()