import gradio as gr
from langchain_ollama import OllamaLLM
from langchain import PromptTemplate, LLMChain
from fpdf import FPDF
import tempfile

# Initialize the Ollama LLM
langchain_llm = OllamaLLM(model="llama3.2")

# Define the prompt template
prompt_template = """
You are a travel assistant. Using the following inputs, create a detailed itinerary:
- Destination: {destination}
- Month: {month}
- Duration: {duration} days
- Number of People: {num_people}
- Holiday Type: {holiday_type}
- Budget Type: {budget_type}
- Additional Comments: {comments}

Ensure the plan suits the specified preferences and includes activities, dining options, and relaxation time. Format the response neatly with sections for each day.
"""

# Set up the chain
prompt = PromptTemplate(
    input_variables=["destination", "month", "duration", "num_people", "holiday_type", "budget_type", "comments"],
    template=prompt_template,
)
chain = LLMChain(llm=langchain_llm, prompt=prompt)

# Global chat history
chat_history = []

# Function to generate itinerary
def plan_trip(destination, month, duration, num_people, holiday_type, budget_type, comments):
    global chat_history
    response = chain.run({
        "destination": destination,
        "month": month,
        "duration": duration,
        "num_people": num_people,
        "holiday_type": holiday_type,
        "budget_type": budget_type,
        "comments": comments,
    })
    formatted_response = f"**Itinerary for {destination} ({holiday_type}) - {budget_type}**\n\n" + response
    chat_history.append(("System", formatted_response))
    return chat_history

# Function to chat with itinerary
def chat_with_itinerary(user_message):
    global chat_history
    context = "\n".join([f"{sender}: {message}" for sender, message in chat_history])
    prompt = f"Context:\n{context}\n\nUser: {user_message}\n\nSystem:"
    response = langchain_llm(prompt)
    chat_history.append(("User", user_message))
    chat_history.append(("System", response))
    return chat_history

# Function to clear chat history
def clear_chat():
    global chat_history
    chat_history = []
    return chat_history

# Function to export chat history as PDF
def export_to_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Ensure text is encoded correctly
    for sender, message in chat_history:
        message = message.encode('latin-1', 'ignore').decode('latin-1')  # Handle non-latin characters
        pdf.multi_cell(0, 10, f"{sender}: {message}\n\n")
    
    # Generate a PDF in a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# UI Design
def interactive_trip_planner():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### AI-RouteRover")
                destination = gr.Textbox(label="City/Country Name", placeholder="Enter destination")
                month = gr.Dropdown(choices=[
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ], label="Month of Travel")
                duration = gr.Slider(minimum=1, maximum=30, label="Number of Days", value=7)
                num_people = gr.Dropdown(choices=["1", "2", "3","4-6" "7-10", "10+"], label="Number of People")
                holiday_type = gr.Dropdown(choices=[
                    "Any", "Party holiday", "Skiing and snowboarding", "Backpacking",
                    "Family holiday", "Beach holiday", "Music festivals",
                    "Adventure holidays", "City break", "Romantic holiday", "Cruise holiday"
                ], label="Holiday Type", value="Any")
                budget_type = gr.Dropdown(choices=[
                    "Budget Travel", "Mid-Range Budget", "Luxury Budget",
                    "Backpacker Budget", "Family Budget"
                ], label="Budget Type")
                comments = gr.Textbox(label="Additional Comments", placeholder="Any specific requirements or preferences?")
                submit_btn = gr.Button("Plan Trip")

            with gr.Column(scale=2):
                chatbox = gr.Chatbot(label="Chat About Your Itinerary")
                chat_input = gr.Textbox(label="Your Message", placeholder="Ask questions or make comments")
                gr.Markdown("####") 
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", size="sm")
                    export_btn = gr.Button("Export as PDF", size="sm")



        # Define button actions
        submit_btn.click(plan_trip, 
                         inputs=[destination, month, duration, num_people, holiday_type, budget_type, comments], 
                         outputs=chatbox)
        chat_input.submit(chat_with_itinerary, inputs=chat_input, outputs=chatbox)
        clear_btn.click(clear_chat, outputs=chatbox)
        export_btn.click(export_to_pdf, outputs=None)

    interface.launch(share=True)

# Launch the UI
interactive_trip_planner()
