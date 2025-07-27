#from src.prompt import attaction_prompt, weather_prompt

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence, List, Dict
from langchain_core.messages import BaseMessage
import operator
#from src.tool_aggregator import attract_activities_tool, weather_tool
from src.llm import initiate_llm, initiate_embeddings
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field
from typing import TypedDict, List, Annotated, Sequence
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import PromptTemplate
from src.data_ingestion import *
from src.config import urls
from langchain_community.document_loaders import WebBaseLoader

def load_docs(urls):
    loader=WebBaseLoader(urls)
    docs = loader.load()
    print(len(docs))
    return docs

llm = initiate_llm()
embeddings = initiate_embeddings()

docs = load_docs(urls)
#vector_store = main()
# Load the existing store
vector_store = Chroma(
    collection_name="puffy_lumous",  # ✅ Must match
    embedding_function=embeddings,
    persist_directory="./puffy_chroma_store"  # ✅ Must match
)
retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(retriever=retriever,
                         name="puffy_lumous",
                          description="Details about mattress, smart bed, bedframes, beddings & more"  + '\n\n'.join([doc.metadata['description'] for doc in docs])
                        )
tools = [retriever_tool]


#retriever = vector_store.as_retriever()
#retriever_tool = create_retriever_tool(retriever=retriever,
#                         name="puffy_lumous",
#                          description="Details about mattress, smart bed, bedframes, beddings & more"  + '\n\n'.join([doc.metadata['description'] for doc in docs])
#                        )
#tools = [retriever_tool]

print("Number of vectors:", vector_store._collection.count())  # Check if DB has vectors
results = vector_store.similarity_search("mattress", k=3)
print(results)  # See if any result is retrieved

class AgentState(TypedDict):
    lang: str
    user_query: str
    documents: List[str]
    messages: List[BaseMessage]
    llm_response: str
    
class QueryExtraction(BaseModel):
    lang: str=Field(description="language of the user question in the standard of ISO 639-1: Example EN for english; FR for french")
    translated: str=Field(defaullt=None, description="English translated query if user has entered non english question")

def supervisor(state:AgentState):
    print('<<--supervisor-->>', state["user_query"])
    llm_with_tools = llm.bind_tools(tools) 
    ai_message = llm_with_tools.invoke(state["user_query"])
    #print(f"response:{ai_message}")
    return {'messages': [ai_message]}

def extractor(state:AgentState):
    print('<<--extractor-->>', state['user_query'])
    
    template = """
    You are a helpful assistant. Analyze the user query. Analyze the query and respond the language code(standard of ISO 639-1) of the query.
    Example EN for english; FR for french. For non english, translate the non-english query to english query.
    if the user is already in english then do nothing, populate empty space to translated field
    
    User_query: {user_query}
    """
    prompt = PromptTemplate(template=template,
                   input_variables=['user_query'])
    print('prompt:', prompt)
    chain = prompt | llm.with_structured_output(QueryExtraction)
    response=chain.invoke({'user_query':state['user_query']})
    
    if response.lang == 'EN':
        final_query = state['user_query']
    else:
        final_query = response.translated
        
    return {
        'lang'       : response.lang,
        'user_query' : final_query
    }

def conditional_router(state:AgentState):
    if state['lang'] == 'EN':
        return 'retriever'
    else:
        return 'retriever'
    
def run_retriever(state:AgentState):
    print('<<--retriever-->>', state['user_query'])
    query = state['user_query']
    if state['lang'] == 'EN':
        docs = retriever.invoke(query[0].content)
    else:
        docs = retriever.invoke(query)
    print(f'Query for retriever: Retreived_docs: {docs}')
    return {"documents":docs} 

def synthesizer(state:AgentState):
    print('<<--synthesizer-->>', state['user_query'])
    template = """
    You are a helpful assistant. Based on the user query regarding Puffy product, I will provide you the context 
    from which you need to generate the response along with the actual user query.
    Follow below instructions
    1) Try to be precise and crisp - No lengthy answer, if possible bulleted points if relevant
    2) recommend the relevant product(Ex: Cloud, Lux, Royal etc) as per the user requirement, 
    3) share the cost for the product,
    4) Share the offer if applicable
    5) tell me importance/significance of the product in one liner
    6) provide the links to read further or buy the product.
    7) Bold the important terms like product name, cost, discount, offer etc
    \n
    User_query: {user_query}
    context   : {documents}
    """
    prompt = PromptTemplate(template=template,
                   input_variables=['user_query', 'documents'])
    print('prompt:', prompt)
    chain = prompt | llm
    response=chain.invoke({'user_query':state['user_query'],
                           'documents' :state['documents']
                               })
    print(response)
    return {'llm_response':response}
    
    
def workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("extractor", extractor)
    workflow.add_node("run_retriever", run_retriever)
    workflow.add_node("synthesizer", synthesizer)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", 
                                tools_condition, 
                                {'tools': "extractor",
                                    END: END
                                    }
                                )
    #workflow.add_edge("supervisor", "extractor")
    workflow.add_conditional_edges("extractor",
                                conditional_router,
                                {
                                    'retriever':'run_retriever'
                                }
                                )
    workflow.add_edge('run_retriever', 'synthesizer')
    workflow.add_edge('synthesizer', END)

    app = workflow.compile()
    return app