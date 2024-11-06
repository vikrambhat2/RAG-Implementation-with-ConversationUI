import streamlit as st
import praw
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Access Reddit API credentials
client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")
user_agent = os.getenv("user_agent")

# Initialize PRAW and summarization/QA pipelines
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

# Initialize LLM with Ollama's model
llm = Ollama(model="llama3.2")

# Set up HuggingFace embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

def ingest_into_vectordb(split_docs):
    """Store split documents in FAISS vector database and save locally."""
    db = FAISS.from_documents(split_docs, embeddings)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print("Documents are inserted into FAISS vectorstore")
    return db

# Define PromptTemplate for summarization and Q&A
prompt_template = PromptTemplate(input_variables=["text"], template="{text}")

# Streamlit UI
st.title("Reddit Summarizer & Q&A Bot")

# Text input for Reddit URL and mode selection
url = st.text_input("Enter Reddit post URL:")
mode = st.sidebar.radio("Choose a mode:", ("Summarize", "Answer Questions"))

def process_reddit_post(url):
    """Fetch and process Reddit post and comments, returning chunked Document objects."""
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=None)
    content = submission.selftext + "\n" + "\n".join([comment.body for comment in submission.comments.list()])

    # Chunk content for FAISS storage using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(content)

    # Create Document objects for the chunks
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Ingest into FAISS vector database
    vectorstore=ingest_into_vectordb(documents)

    return vectorstore

if url:
    # Process Reddit post if URL is provided
    vector_store = process_reddit_post(url)

    if mode == "Summarize":
        # Retrieve top chunks for summarization
        query = "Summarize this content."
        relevant_docs = vector_store.similarity_search(query, k=5)
        context = " ".join([doc.page_content for doc in relevant_docs])

        context ="Here are the comments on a reddit post\n" + context

        # Chain prompt with LLM using RunnableSequence
        summary = (prompt_template | llm).invoke(context)
        st.write("### Summary")
        st.write(summary)

    elif mode == "Answer Questions":
        # User input for question
        question = st.text_input("Enter your question:")
        if question:
            # Retrieve top chunks for Q&A
            relevant_docs = vector_store.similarity_search(question, k=5)
            context = " ".join([doc.page_content for doc in relevant_docs])
            

            # Prepare the input for the question prompt template
            question_template = PromptTemplate(
                input_variables=["text", "question"],
                template="Here are the comments on a reddit post\n Answer the question based on context: {text}. Question: {question}"
            )
            input_data = {"text": context, "question": question}  # Create a dictionary for input

            # Chain prompt with LLM using RunnableSequence for answering
            answer = (question_template | llm).invoke(input_data)
            st.write("### Answer")
            st.write(answer)
