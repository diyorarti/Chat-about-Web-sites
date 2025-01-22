import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Ensure USER_AGENT is set
os.environ["USER_AGENT"] = "streamlit-app"

# Validate OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY is missing. Please set it in the .env file.")

# Function to get vectorstore from URL
def get_vectorstore_from_url(url):
    from chromadb.config import Settings

    # Get the document from the website
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create a vectorstore from the chunks
    vectore_store = Chroma.from_documents(
        document_chunks,
        OpenAIEmbeddings(),
        persist_directory="./chroma_data"  # Specify directory for persistence
    )
    return vectore_store

# Function to get context retriever chain
def get_context_retriever_chain(vectore_store):
    llm = ChatOpenAI()

    retriever = vectore_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# Function to get conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Function to get response
def get_response(user_input):
    try:
        retriever_chain = get_context_retriever_chain(st.session_state.vectore_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })

        return response['answer']
    except Exception as e:
        st.error(f"Error processing response: {e}")
        return "Sorry, I encountered an error while processing your request."

# Initialize Streamlit interface
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# Check website URL input
if website_url is None or website_url == "":
    st.info("Please enter a website URL.")
else:
    # Session state initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, how can I help you?")
        ]
    if "vectore_store" not in st.session_state:
        try:
            st.session_state.vectore_store = get_vectorstore_from_url(website_url)
        except Exception as e:
            st.error(f"Error initializing vectorstore: {e}")
            st.stop()

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
