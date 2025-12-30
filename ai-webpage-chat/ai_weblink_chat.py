import streamlit as st

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the answer is not in the context, say you don't know and suggest what the user can do next.
Use concise, clear language.

Context: {context}
Question: {question}

Answer:

"""

embeddings = OllamaEmbeddings(model="llama3.2")  #converts our text to vector
vector_store = InMemoryVectorStore(embeddings)


model = OllamaLLM(model="llama3.2")


#1. User provides URL, the site is loaded and text is extracted from it
def load_page(url):
    loader = SeleniumURLLoader( 
        urls = [url]     
      )  #this loads the webpage dynamic content inside the browser
    
    documents = loader.load()  #this gives a document wrapper of the webpage used by langchain
    return documents

#2. The site text is splitted into smaller documents because LLMs cannot deal with large
# text efficiently
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index= True
    )
    data = text_splitter.split_documents(documents)   # gives a list of docs
    return data


#3. The documents are indexed by doing embeddings on it and coverted to vector 
# and stored inside the vector store
def index_docs(documents):
    vector_store.add_documents(documents)  #langchain gives the docs to Ollama's embedding model and store the vectors 

#4. After this, the web info can be fetched quickly
def retrieve_docs(query):    #pass user query to vector store and return related docs
    return vector_store.similarity_search(query)


#5. Here the user questions are answered. When question is received,
#  it is passed to the vector store and all documents related to the question
# are retrieved, also a prompt is provided to the LLM to answer only questions 
# based on the information provided to it. User questions and retrieved doc is passed
# to the prompt, with this technique we should get very good LLM response.
def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question":question, "context":context}) 

st.title("AI Webpage Chat:")
url =st.text_input("Enter web URL:")

documents = load_page(url)
chunked_documents = split_text(documents)

index_docs(chunked_documents)
question = st.chat_input()

#Show users question
if question:

    st.chat_message("User").write(question)
    retrieve_documents = retrieve_docs(question)
    context = "\n\n".join([doc.page_content for doc in retrieve_documents])
    answer = answer_question(question, context)
    st.chat_message("Assistant").write(answer)