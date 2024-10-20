import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def get_pdf(pdfs):
    """Extracts text from PDF files."""
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)

        for page in reader.pages:
            text += page.extract_text()

    return text


def text_chunks(text):
    """Splits text into chunks for vectorization."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=1000)  # Updated chunk_size
    chunks = splitter.split_text(text)
    return chunks


def vectorize(chunks):
    """Creates a FAISS vector store from the text chunks."""
    faiss_embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectors = FAISS.from_texts(chunks, embedding=faiss_embeddings)
    vectors.save_local("faiss_index")


def load_faiss_index():
    """Loads the FAISS index from a local file."""
    faiss_embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local("faiss_index", embeddings=faiss_embeddings, allow_dangerous_deserialization=True)
    return new_db


def conversation():
    """Creates a conversational chain."""
    template = """Answer the question in detail from the given context. If the answer is not in the data,
    just say, "Answer is not available in the uploaded document".

    context = {context}
    question = {question} """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create the LLM chain by combining the model and the prompt
    llm_chain = LLMChain(llm=model, prompt=prompt)

    # Create the StuffDocumentsChain and specify document_variable_name
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    return chain


# def user_input(question):
#     """Handles user input and provides a response."""
#     new_db = load_faiss_index()
#     if not new_db:
#         return

#     # Perform a similarity search in FAISS index
#     docs = new_db.similarity_search(question)

#     chain = conversation()

#     # Run the QA chain and get the response
#     response = chain({
#         "context": docs,
#         "question": question
#     })

#     st.write("Reply: ", response["output_text"])


def user_input(question):
    """Handles user input and provides a response."""
    new_db = load_faiss_index()
    if not new_db:
        st.error("No document database found. Please upload a PDF first.")
        return

    # Perform a similarity search in FAISS index
    docs = new_db.similarity_search(question)
    
    if not docs:
        st.error("No relevant documents found.")
        return

    chain = conversation()

    # Prepare input for the chain, use 'input_documents' for the context
    response = chain({
        "input_documents": docs,
        "question": question
    })

    st.write("Reply: ", response["output_text"])



def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF")

    # Text input for user question
    question = st.text_input("Ask your question...")

    # If question is provided, handle user input
    if question:
        user_input(question)

    # Sidebar for PDF file upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload the PDF file", type="pdf", accept_multiple_files=True)
        if pdf_docs and st.button("Submit"):
            raw_text = get_pdf(pdf_docs)  # Extract text from PDFs
            chunks = text_chunks(raw_text)  # Split text into chunks
            vectorize(chunks)  # Vectorize chunks and save FAISS index
            st.success("PDF processed and vectorized successfully!")


if __name__ == "__main__":
    main()
