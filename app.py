import streamlit as st
from helpers import get_api_key, process_pdfs
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

def main():
    """ Variable to render web app functionality based on certain conditions """
    disable_input = False

    # Create streamlit web app layout

    # Basic Page Config
    st.set_page_config(page_title="PDF Chat bot")
    st.header("PDF Chat bot")

    # Check for API Key
    if not get_api_key():
        st.error("No OPENAI_API_KEY set. Please set environment variable and restart app to enable.")
        disable_input = True

    # Create File Uploader
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"], disabled=disable_input)

    if uploaded_files:
        knowledge_base = process_pdfs(uploaded_files)
        st.success(f"Uploaded {len(uploaded_files)} files")

    # Create Text Input
    text_input = st.text_input("Ask a question about your PDFs", disabled=disable_input)

    if text_input:
        relevant_docs = knowledge_base.similarity_search(text_input)
        
    # Process Output
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=relevant_docs, question=text_input)

    st.write(response)

if __name__ == "__main__":
    main()