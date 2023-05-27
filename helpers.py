from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def get_api_key():
    """
    Loads API key from .env file

    Returns
    -------
    str
        OPEN AI API key
    """

    load_dotenv()
    return os.getenv("OPENAI_API_KEY")

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
        text += page.extractText()
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
    pdf_texts : list
        List of text from pdfs
    """

    pdf_texts = []
    for pdf in pdfs:
        pdf_texts.append(read_pdf(pdf))
    return pdf_texts

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
        seperator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split(text=pdf_text)

    return chunks

def create_all_chunks(pdf_texts):
    """
    Creates chunks for all PDFs together

    Parameters
    ----------
    pdf_texts : list
        List of text from pdfs

    Returns
    -------
    all_chunks : list
        List of text chunks
    """

    all_chunks = []
    for pdf_text in pdf_texts:
        all_chunks.append(make_chunks(pdf_text))

    return all_chunks

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
    knowledge_base = FAISS.from_texts(all_chunks, embeddings=embeddings)
    
    return knowledge_base

def process_pdfs(pdfs):
    """
    Processes PDFs to create knowledge base

    Parameters
    ----------
    pdfs : list
        List of pdf files

    Returns
    -------
    knowledge_base 
        FAISS embedding generated from all chunks
    """

    pdf_texts = extract_pdfs(pdfs)
    all_chunks = create_all_chunks(pdf_texts)
    knowledge_base = create_embeddings(all_chunks)

    return knowledge_base