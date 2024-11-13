import pandas as pd
import gradio as gr
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama.llms import OllamaLLM

# Initialize the LLM
langchain_llm = OllamaLLM(model="llama3.2")

# Define a function to create the agent and answer questions based on the uploaded CSV
def query_dataframe(file, question, history):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file.name)
    template = """
        Please provide the answer within 10 words from the provided data only.
    """
    
    # Create the agent with the DataFrame and set allow_dangerous_code=True
    p_agent = create_pandas_dataframe_agent(
        llm=langchain_llm,
        df=df,
        prompt_template=template,
        verbose=False,
        max_iterations=500,            # Limit to 5 iterations, or adjust as needed
        max_execution_time=1000,
        allow_dangerous_code=True # Enables handling of parsing errors
    )
    
    # Run the agent with the input question and capture the response
    try:
        response = p_agent.run(question)
    except ValueError as e:
        response = f"An error occurred: {e}"  # Handle any parsing error gracefully
    
    # Append the new question-response pair to the chat history
    history.append((question, response))
    return history, history  # Return both the updated chat history for display and for state

# Define the Gradio UI with a chat history
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        <style>
        .gradio-container {
            background-color: #fffbe7;
            font-family: Arial, sans-serif;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            text-align: center;
            padding: 10px;
        }
        </style>
        """
    )

    gr.Markdown("<div class='header'>CSV Query Interface</div>")

    # CSV file uploader
    csv_file = gr.File(label="Upload CSV file", file_types=[".csv"])
    
    # Text box for entering the question
    question = gr.Textbox(label="Enter your question about the CSV data", placeholder="Type your question here...")
    
    # Button to submit the question
    query_button = gr.Button("Submit", variant="primary")
    
    # Chatbot for displaying responses in a chat format
    response_display = gr.Chatbot(label="Chat History")
    
    # Initialize an empty history variable
    history = gr.State([])

    # Function to update the chatbot display with the model's answer
    def update_response(file, question, history):
        return query_dataframe(file, question, history)

    # Bind the update_response function to the Submit button
    query_button.click(update_response, inputs=[csv_file, question, history], outputs=[response_display, history])

# Launch the app
if __name__ == "__main__":
    app.launch()
