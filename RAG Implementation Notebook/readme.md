

# Conversational RAG Application with Ollama and Llama 3.2

`Implement RAG using Llama 3.2.ipynb` notebook implements a Conversational Retrieval-Augmented Generation (RAG) application using Ollama and the Llama 3.2 model. The application allows for efficient document loading, splitting, embedding, and conversation management. Below are detailed descriptions of the key functions and installation instructions for Ollama.


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
