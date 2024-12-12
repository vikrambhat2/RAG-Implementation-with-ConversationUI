from dotenv import load_dotenv
import os
from crewai import Agent, Crew, Task, LLM, Process
import requests
from agents.web_crawler_agent import WebCrawlerAgent
from agents.summarizer_agent import SummarizerAgent
from agents.report_generator_agent import ReportGeneratorAgent
import gradio as gr

load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Create an LLM instance for Ollama LLaMA 3.2
llama_model = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"  # URL for Ollama's local server
)



# Initialize the agents
web_crawler_agent = WebCrawlerAgent(
    llm=llama_model,
    role="web crawler",
    backstory="I search the web to gather information on a given topic.",
    goal="Fetch web search results for a given query.",
    serper_api_key=SERPER_API_KEY
)


summarizer_agent = SummarizerAgent(llama_model)

report_generator_agent = ReportGeneratorAgent(llm=llama_model)



# Modified generate_report function to include report style
def generate_report(query, report_style):
    # Task definitions
    web_crawler_task = Task(
        description=query, 
        expected_output="A detailed response about the query.",
        agent=web_crawler_agent
    )

    summarizer_task = Task(
        description="Summarize the web data.", 
        expected_output="Concise summaries of the search results.", 
        agent=summarizer_agent
    )

    report_task = Task(
        description=f"Generate a final research report in '{report_style}' style by removing bias if any.",
        expected_output="A comprehensive research report.",
        agent=report_generator_agent
    )

    # Create the Crew
    crew = Crew(
        agents=[web_crawler_agent, summarizer_agent, report_generator_agent],
        tasks=[web_crawler_task, summarizer_task, report_task],
        verbose=True,
        process=Process.sequential,
        context_flow={
            "summarizer_task": "web_crawler_task",  # Output of web_crawler_task goes to summarizer_task
            "report_task": "summarizer_task",      # Output of summarizer_task goes to report_task
        }
    )

    # Execute tasks and return the final output
    result = crew.kickoff()
    formatted_report = f"###### Report Generated in {report_style} Style:\n\n{result.raw}"

    # Return the formatted report
    return formatted_report

# Gradio Interface
with gr.Blocks() as demo:
    # Improved visuals using markdown and layout changes
    gr.Markdown("## 🚀 Query-Based Research Report Generator")
    gr.Markdown("#### Enter your query and choose the report style to generate a detailed, formatted report!")

    with gr.Row():
        query_input = gr.Textbox(
            label="Enter Your Query", 
            placeholder="e.g., Recent advancements in Indian Agriculture",
            lines=2,
            elem_id="query-input"
        )

    with gr.Row():
        report_style = gr.Radio(
            choices=[ "Detailed Analysis", "Brief Summary","Bullet Points"],
            label="Select Report Style",
            value="Detailed Analysis"
        )
    
    with gr.Row():
        generate_button = gr.Button("Generate Report", elem_id="generate-button", variant="primary")
    
    with gr.Row():
        report_output = gr.Markdown(
            label="Generated Report"
        )
    
    generate_button.click(
        generate_report, 
        inputs=[query_input, report_style], 
        outputs=[report_output]
    )

# Launch the Gradio app
demo.launch(share=True)
