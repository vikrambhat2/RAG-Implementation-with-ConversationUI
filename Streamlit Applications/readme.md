Streamlit app, `StreamlitApp_V2.py`, explaining how to execute it:

# Document Chatbot Application

This Streamlit application allows users to interact with their PDF documents in a chat format. Users can ask questions, view bot responses, and toggle the display of related source documents. The app also features an answer relevancy score to assess the accuracy of the bot's responses.

## Features

- **Chat Interface:** Users can input questions and receive answers from the bot based on uploaded PDF documents.
- **Show/Hide Source Documents:** For each bot response, users can toggle the display of relevant source documents.
- **Answer Relevancy Score:** Users can calculate a similarity score between the bot's answer and the source documents.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- LangChain
- Sentence Transformers
- Hugging Face Transformers

## Installation

1. Clone the repository or download the project files.
2. Navigate to the project directory in your terminal.
3. Install the required packages using pip:

   ```bash
   pip install streamlit langchain sentence-transformers huggingface_hub
   ```

## Running the Application

1. Open a terminal or command prompt.
2. Navigate to the directory where your `StreamlitApp_V2.py` file is located.
3. Run the application with the following command:

   ```bash
   streamlit run StreamlitApp_V2.py
   ```

4. Once the server starts, open your web browser and go to `http://localhost:8501` to access the application.

## Usage

- Upload PDF documents using the sidebar.
- Click "Process PDFs" to prepare the documents for querying.
- Type your question in the input field and click "Submit" to receive an answer from the bot.
- Use the "Show/Hide Source Docs" button to toggle the display of related source documents.
- Click "Calculate Answer Relevancy" to see the similarity score of the bot's answer against the source documents.

## Contributing

Contributions and suggestions are welcome! Feel free to open an issue or submit a pull request if you have ideas for improvements or new features.

![image](https://github.com/vikrambhat2/RAG-using-elasticsearch-WatsonX/assets/90188807/1f1e1821-8695-4e9b-a93b-47f52733e46a)
