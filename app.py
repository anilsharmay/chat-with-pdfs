import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import time # Import time for potential small delays to visualize progress better

# Load environment variables from .env file (for local development)
load_dotenv()

# --- Configuration & API Key Check ---
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not found. Please set it as an environment variable (e.g., in a .env file locally, or in Streamlit Cloud secrets).")
    st.stop()

# --- Functions for PDF Processing and Chat Logic ---

@st.cache_resource
def process_pdf(pdf_docs):
    """
    Extracts text from PDF documents, chunks it, creates embeddings using Gemini,
    and stores them in an in-memory Chroma vector store, with progress updates.
    """
    # Use st.empty() and st.progress() to update messages and the bar in place
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    try:
        status_placeholder.info("1/4: Starting PDF processing...")
        progress_bar.progress(0)

        raw_text = ""
        total_pages = 0
        
        # First pass: Count total pages to set up accurate progress bar
        for pdf_doc in pdf_docs:
            pdf_reader_count = PdfReader(pdf_doc)
            total_pages += len(pdf_reader_count.pages)
        
        if total_pages == 0:
            status_placeholder.error("No pages found in the uploaded PDF(s).")
            progress_bar.empty()
            return None

        # Second pass: Extract text and update progress
        current_page_processed = 0
        for pdf_doc in pdf_docs:
            pdf_reader = PdfReader(pdf_doc)
            for i, page in enumerate(pdf_reader.pages):
                raw_text += page.extract_text()
                current_page_processed += 1
                page_progress = current_page_processed / total_pages
                progress_bar.progress(0.01 + page_progress * 0.29) # 1% to 30% for extraction
                status_placeholder.info(f"2/4: Extracting text from page {current_page_processed} of {total_pages}...")
                # Optional: time.sleep(0.01) if progress bar updates too fast for small files

        if not raw_text.strip():
            status_placeholder.error("Could not extract any meaningful text from the PDF(s). They might be scanned images.")
            progress_bar.empty()
            return None
        
        status_placeholder.info(f"2/4: Text extraction complete. Total characters: {len(raw_text):,}")
        progress_bar.progress(0.30)

        status_placeholder.info("3/4: Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        
        if not chunks:
            status_placeholder.error("Text splitting resulted in no chunks. The document might be too short or unusual.")
            progress_bar.empty()
            return None

        status_placeholder.info(f"3/4: Text split into {len(chunks):,} chunks. Preparing for embeddings...")
        progress_bar.progress(0.60)

        status_placeholder.info("4/4: Generating embeddings and building vector store (this is often the longest step)...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
        
        status_placeholder.success("PDF(s) processed and ready to chat!")
        progress_bar.progress(1.0)
        return vectorstore

    except Exception as e:
        status_placeholder.error(f"An error occurred during PDF processing: {e}")
        progress_bar.empty() # Clear the progress bar on error
        return None

def get_conversation_chain(vectorstore):
    """
    Initializes and returns a conversational retrieval chain using Gemini 2.5 Pro.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.5, convert_system_message_to_human=True) 
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- Streamlit User Interface (UI) ---

st.set_page_config(page_title="PDF Chatbot (Gemini 2.5 Pro)", page_icon="📄", layout="centered")

st.header("Chat with your PDF(s) 📄 using Gemini 2.5 Pro")

# Sidebar for PDF upload and processing button
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click 'Process'", accept_multiple_files=True, type=["pdf"]
    )
    if st.button("Process PDFs"):
        if pdf_docs:
            # The spinner is now managed inside process_pdf with more granular updates
            st.session_state.clear() # Clear session state to ensure fresh processing
            vectorstore = process_pdf(pdf_docs)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation_chain = get_conversation_chain(vectorstore)
                st.session_state.pdf_processed = True
                st.experimental_rerun() # Rerun to refresh the main chat area
            else:
                st.error("PDF processing failed. Please try again with valid PDF files.")
        else:
            st.warning("Please upload at least one PDF file.")

# Main chat area initialization and logic
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.messages = []

if not st.session_state.pdf_processed:
    st.warning("Please upload and process PDFs in the sidebar to start chatting with the bot.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your PDF:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.conversation_chain:
                    try:
                        response = st.session_state.conversation_chain.invoke({"question": prompt})
                        st.markdown(response['answer'])
                        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                    except Exception as e:
                        st.error(f"An error occurred while generating a response: {e}. Please try again or re-process the PDF.")
                        st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
                else:
                    st.error("Error: Conversation chain not initialized. Please re-process the PDF.")