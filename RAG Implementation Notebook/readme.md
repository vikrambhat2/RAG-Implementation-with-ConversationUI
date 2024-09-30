

# Conversational RAG Application with Ollama and Llama 3.2

`Implement RAG using Llama 3.2.ipynb` notebook implements a Conversational Retrieval-Augmented Generation (RAG) application using Ollama and the Llama 3.2 model. The application allows for efficient document loading, splitting, embedding, and conversation management. Below are detailed descriptions of the key functions and installation instructions for Ollama.

## Function Summaries

### 1. `prepare_and_split_docs`
This function loads PDF documents from the specified directory and splits them into manageable chunks. It utilizes the `DirectoryLoader` to fetch PDFs and employs the `RecursiveCharacterTextSplitter` to create text passages of a specified size, ensuring metadata is preserved. The resulting split documents facilitate efficient processing for retrieval and response generation.

### 2. `ingest_into_vectordb`
This function takes the split documents and ingests them into a vector database using embeddings. It utilizes the `HuggingFaceEmbeddings` class to create embeddings from the documents and stores them in a FAISS vector store. The database is saved locally for efficient retrieval in subsequent queries.

### 3. `get_conversation_chain`
This function sets up the conversation chain, integrating the Ollama model to process user queries and retrieve relevant document context. It establishes a history-aware retriever using the specified chat prompt and creates a retrieval chain for answering questions based on the document context. The function maintains chat history for contextually aware responses.

### 4. `calculate_similarity_score`
This function computes the similarity score between a generated answer and the provided context documents. It employs the `SentenceTransformer` to encode both the answer and context documents, then calculates cosine similarities to determine the most relevant context. The maximum similarity score is returned for evaluation.

## Installation Instructions for Ollama and Llama 3.2

Follow these steps to install and run Ollama locally:

1. **Download Ollama**  
   Visit the following link to download Ollama:  
   [Download Ollama](https://ollama.com/download)

2. **Extract the Zip File**  
   Once downloaded, extract the zip file and locate the Ollama executable.

3. **Run Ollama**  
   After installation, the Ollama icon will appear in your taskbar. This indicates that the Ollama server is up and running, listening on HTTP port **11434** of your localhost. You can confirm the server status by visiting the following URL:  
   [http://localhost:11434/](http://localhost:11434/)  
   You should see a message indicating that **Ollama is running**.

4. **Open Command Prompt**  
   Launch a Windows terminal (Command Prompt) and execute the following command to run the Llama-3 model locally:  
   ```bash
   ollama run llama3.2
   ```

5. **Import Ollama in Your Code**  
   In your Python script, import the Ollama class and create an instance for the Llama-3 model:  
   ```python
   from langchain_community.llms import Ollama
   llm = Ollama(model="llama3.2")
   ```


```
