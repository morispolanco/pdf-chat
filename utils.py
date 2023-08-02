import openai
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template, user_template
from tiktoken import tiktoken


openai.api_key = os.environ.get("OPENAI_API_KEY")

def read_pdf(pdf):
    """
    Reads and returns complete text from singluar PDF (helper fnc)

    Parameters
    ----------
    pdf 
        PDF file to extract

    Returns
    -------
    text : str
        Complete text from PDF
    """

    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_pdfs(pdfs):
    """
    Reads and extracts text from PDFs

    Parameters
    ----------
    pdfs : list
        List of pdf files

    Returns
    -------
    pdf_texts : str
        Complete text from pdfs
    """

    pdf_text = ""
    for pdf in pdfs:
        pdf_text += read_pdf(pdf)
    return pdf_text

def make_chunks(pdf_text):
    """
    Creates chunks of text from PDF

    Parameters
    ----------
    pdf_text : str
        Text from PDF

    Returns
    -------
    chunks : list
        List of text chunks
    """

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text=pdf_text)

    return chunks

def create_embeddings(all_chunks):
    """
    Creates embeddings for all chunks

    Parameters
    ----------
    all_chunks : list
        List of text chunks

    Returns
    -------
    knowledge_base 
        FAISS embedding generated from all chunks
    """

    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings)
    
    return vector_store

def get_conversation_chain(embeddings):
    """
    Creates conversation chain

    Parameters
    ----------
    embeddings : List
        Embeddings from all chunks

    Returns

    -------
    conversation_chain : 
        Conversation chain
    """
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=embeddings.as_retriever(),
        memory=memory,
    )

    return conversation_chain

def handle_input(question):
    """
    Handles user input

    Parameters
    ----------
    question : str
        User input

    Returns
    -------
    bot_response : str
        Response from bot
    """

    bot_response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = bot_response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
