import streamlit as st
import tiktoken
import openai
import os
from utils import extract_pdfs, make_chunks, create_embeddings, get_conversation_chain, handle_input
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from htmlTemplates import css, bot_template, user_template

def handle_input(question):
    # Get the chatbot response
    response = st.session_state.conversation

    if st.session_state.chat_history is not None:
        # reverse the chat history before enumeration
        reversed_chat_history = reversed(st.session_state.chat_history)

        for i, message in enumerate(reversed_chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{MSG}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{MSG}", message.content), unsafe_allow_html=True)

def handle_input(question):
    # Get the chatbot response
    response = st.session_state.conversation
    print('DEBUG: Response:', response)

    if st.session_state.chat_history is not None:
        # debug the chat history 
        print('DEBUG: Chat history:', st.session_state.chat_history)
        
def main():
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Create streamlit web app layout
    st.set_page_config(page_title="PDF Chat bot")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF Chat bot")
    # Basic Page Config
    question = st.text_input("Ask a question about your PDFs")

    if question:
        handle_input(question)

    # Create Sidebar
    with st.sidebar:
        st.subheader("Add PDFs")
        docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Upload"):
            with st.spinner("Processing"):

                # Process PDFs
                raw_text = extract_pdfs(docs)

                # Create Text Chunks
                text_chunks = make_chunks(raw_text)

                # Create Embeddings
                embeddings = create_embeddings(text_chunks)

                # Conversations with LLMs
                st.session_state.conversation = get_conversation_chain(embeddings)


if __name__ == "__main__":
    main()
