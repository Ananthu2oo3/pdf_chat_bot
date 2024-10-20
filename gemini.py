import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import StuffDocumentsChain

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=80000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    print(chunks)
    return chunks


def vectorize(chunks):
    """Creates a FAISS vector store from the text chunks."""
    faiss_embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    # vectors = FAISS.from_texts(chunks, embedding=faiss_embeddings)
    vectors = FAISS.from_texts(chunks, embedding=faiss_embeddings)
    vectors.save_local("faiss_index")


def load_faiss_index():
    """Loads the FAISS index from a local file."""
    faiss_embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local("faiss_index", embeddings=faiss_embeddings, allow_dangerous_deserialization=True)
    
    return new_db


# def conversation():
#     """Creates a conversational chain."""
#     template = """Answer the question in detail from the given context. If the answer is not in the data,
#     just say, "Answer is not available in the uploaded document".

#     context = {context}? \n
#     question = {question} """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_name="stuff", prompt=prompt)

#     return chain

# def conversation():
#     """Creates a conversational chain."""
#     template = """Answer the question in detail from the given context. If the answer is not in the data,
#     just say, "Answer is not available in the uploaded document".

#     context = {context}? \n
#     question = {question} """

#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     # chain = load_qa_chain(model, chain_name="stuff", prompt=prompt)
#     chain = StuffDocumentsChain(llm=model, prompt=prompt)

#     return chain


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def conversation():
    """Creates a conversational chain."""
    template = """Answer the question in detail from the given context. If the answer is not in the data,
    just say, "Answer is not available in the uploaded document".

    context = {context}? \n
    question = {question} """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create the LLM chain by combining the model and the prompt
    llm_chain = LLMChain(llm=model, prompt=prompt)

    # Create the StuffDocumentsChain and specify document_variable_name
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    return chain



def user_input(question):
    """Handles user input and provides a response."""
    new_db = load_faiss_index()
    if not new_db:
        return

    docs = new_db.similarity_search(question)

    chain = conversation()

    response = chain({
        "context": docs,
        "question": question
    }, )

    print(response)

    st.write("Reply: ", response)


def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF")

    question = st.text_input("Ask your question...")

    if question:
        user_input(question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload the pdf file", type="pdf", accept_multiple_files=True)
        if pdf_docs and st.button("Submit"):
            raw_text = get_pdf(pdf_docs)
            chunks = text_chunks(raw_text)
            vectorize(chunks)
            st.success("Done")


if __name__ == "__main__":
    main()


    