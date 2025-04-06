import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader

load_dotenv()

def run_doc_chat():
    st.title("📄 Chat with our documentation")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file.")
        st.stop()

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split document
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # Create vectorstore
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=google_api_key
        )
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Gemini Chat LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", temperature=0.7, google_api_key=google_api_key
        )

        # QA Chain
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        st.success("✅ PDF uploaded and processed successfully. Start asking questions!")

    
    # If QA chain is ready, show chat interface
    if st.session_state.qa_chain:
        user_input = st.text_input("Ask a question about the PDF:")
        if user_input:
            with st.spinner("Generating response..."):
                response = st.session_state.qa_chain.run(user_input)
                st.session_state.chat_history.insert(0, {"question": user_input, "answer": response})

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Gemini:** {chat['answer']}")
            st.markdown("---")
