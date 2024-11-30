
# AI-Powered Trip Planner

This project is an AI-powered trip planner built using LangChain, Gradio, and the Ollama LLM model. It allows users to plan their trips by providing a detailed itinerary based on user inputs like destination, holiday type, budget, and other preferences. The system also includes features for chatting with the itinerary and exporting it to a PDF.

## Requirements

Before running the project, make sure you have the following installed:

- **Python 3.7+** (preferably 3.8 or later)
- **Ollama**: You must have the Ollama application installed and running locally. Please follow the [installation instructions here](https://ollama.com/).

### Install Dependencies

To set up the project locally, follow these steps:

1. Clone the repository:


2. Create a virtual environment (optional but recommended):

3. Install the required dependencies using `pip`:

    ```
    pip install -r requirements.txt
    ```

### Requirements in `requirements.txt`

The following packages are required:

- `gradio`: For creating the user interface.
- `langchain`: For building the AI-powered chain and model.
- `langchain-ollama`: For using the Ollama LLM.
- `fpdf`: For exporting the chat history and itinerary to PDF.
- `openai`: For using the OpenAI API (if needed).

## Setting Up Ollama

The project uses **Ollama**, an AI language model running locally, which needs to be installed on your machine.

1. **Install Ollama**:
   - You can download and install the Ollama application from [Ollama's official website](https://ollama.com/).
   - Once installed, make sure the Ollama application is running and accessible on your system.

2. **Ensure the correct model is available**:
   The project uses the `llama3.2` model. Ensure that this model is available in your Ollama installation.

## Running the Gradio UI

1. After setting up the project and installing dependencies, you can run the Gradio interface with the following command:

    ```
    python trip_planner.py
    ```

2. Once the application is running, Gradio will provide a local link (typically something like `http://127.0.0.1:7860`), which you can open in your web browser to interact with the trip planner.

### Features

- **Trip Planning**: Users can enter destination, travel month, number of days, number of people, holiday type, budget type, and additional comments to generate a detailed trip itinerary.
- **Chat about Itinerary**: Users can interact with the generated itinerary by asking questions or leaving comments.
- **Export to PDF**: After planning the trip, users can export the itinerary as a PDF.
- **Clear Chat History**: Option to clear the chat history.

