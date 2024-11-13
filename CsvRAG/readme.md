
# CSV Query Chatbot with LangChain and Ollama

This project demonstrates how to build an interactive CSV query chatbot using **Gradio**, **LangChain's `create_pandas_dataframe_agent`**, and **Ollama's Llama 3.2 model**. The app allows users to upload CSV files and ask natural language questions to analyze the data, without needing specialized skills or complex setups.

## Features

- **CSV File Upload**: Upload any CSV file directly through the Gradio interface.
- **Question Answering**: Ask questions in natural language about the CSV data, and get responses powered by the Llama 3.2 model.
- **Local Execution**: Run the entire application on your local machine, with no need for external servers or APIs.
- **Chat History**: Keep track of all previous questions and answers in a conversational format.

## Prerequisites

- Python 3.7 or higher
- Install the required Python packages:
  - `gradio`
  - `langchain-experimental`
  - `langchain-ollama`
  - `pandas`

## Installation

1. Clone this repository and navigate to this folder:

2. Create and activate a virtual environment (optional)

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure you have Ollama installed and running on your local machine. You can find installation instructions for Ollama [here](https://ollama.com/).

2. Run the Gradio app:
   ```
   python csv_query_app.py
   ```

3. Once the app starts, open your browser and go to the provided URL (usually `http://127.0.0.1:7860/`).

4. Upload your CSV file, type your questions in the textbox, and interact with the chatbot to get insights from the data.

## How It Works

- **CSV Upload**: The user uploads a CSV file which is loaded into a Pandas DataFrame.
- **Question Answering**: The `create_pandas_dataframe_agent` from LangChain interacts with the Llama 3.2 model to answer questions about the data. The model generates human-like responses based on the CSV contents.
- **Chat History**: Each question and response is appended to the chat history for continuous interaction.

## Next Steps

- **RAG Integration**: For larger datasets and more complex queries, consider integrating a **Retrieval-Augmented Generation (RAG)** solution by storing the CSV data in a vector database.
