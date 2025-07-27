from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

def initiate_llm():
    llm = ChatOpenAI()
    return llm

def initiate_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings
