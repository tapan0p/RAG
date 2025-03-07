import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import streamlit as st

def replace_t_with_space(documents):
    processed_docs = []
    for doc in documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
        processed_docs.append(doc)
    return processed_docs

def encode_pdf(path, chunk_size=2000, chunk_overlap=200):
    """ 
    This method will chunk a pdf and then convert the chunks into embedding and store them into a vector database

    Args:
        path: path to the pdf file
        chunk_size: paragraph length of each chunk
        chunk_overlap: max amount of consecutive overlap between chunks

    Return:
        A FAISS vector store containing the encoded pdf content
    """
    # Load pdf documents
    loader = PyPDFLoader(path, mode='page')
    docs = loader.load()

    # Load pdf documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    texts = text_splitter.split_documents(docs)
    texts = replace_t_with_space(texts)
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Define the document path
doc_path = "harry_potter_1.pdf"
vectorstore = encode_pdf(doc_path)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define the prompt template
prompt_template = """Use the following context to answer the question. If you don’t know, say so.
Context: {context}
Question: {question}
Answer: """

# Initialize the LLM
llm = Ollama(model="qwen2.5:3b")

# Create the prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize the retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,  # Fix typo from 'retriver' to 'retriever'
    chain_type_kwargs={"prompt": prompt}
)

def chat(query):
    result = qa_chain({"query": query})
    return result["result"]

def main():
    st.title("QA Chatbot with Ollama")
    st.write("Ask any question based on the retrieved context.")
    
    # User input
    query = st.text_input("Enter your question:")
    if st.button("Ask"):
        if query:
            response = chat(query)
            st.write("### Answer:")
            st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
