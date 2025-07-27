from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from src.llm import initiate_embeddings
from src.config import urls

def load_docs(urls):
    loader=WebBaseLoader(urls)
    docs = loader.load()
    print(len(docs))
    return docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)
    print(len(split_docs))
    print(split_docs[10].page_content)
    return split_docs

def ingest_docs(embeddings, split_docs):
    vector_store = Chroma.from_documents(
                documents=split_docs,
                collection_name="puffy_lumous",
                embedding=embeddings,
                persist_directory="./puffy_chroma_store" 
                )
    vector_store.persist()
    return vector_store

def main():
    docs = load_docs(urls)
    split_docs = split_text(docs)
    embeddings = initiate_embeddings()
    vector_store = ingest_docs(embeddings, split_docs)
    return vector_store
    
if __name__ == "__main__":
    docs = load_docs(urls)
    split_docs = split_text(docs)
    embeddings = initiate_embeddings()
    vector_store = ingest_docs(embeddings, split_docs)