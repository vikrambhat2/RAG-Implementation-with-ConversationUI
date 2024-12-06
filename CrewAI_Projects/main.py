import os
import sys
import argparse
from agents.web_crawler_agent import WebCrawlerAgent
from agents.summarizer_agent import SummarizerAgent
from agents.report_generator_agent import ReportGeneratorAgent
from crewai import Crew, Task, Process
from crewai import LLM

# Read the Serper API key from the environment variable
serper_api_key = "04f7de9cfae51d40e94f1e328d1aa756ffbb14ff" # os.getenv("SERPER_API_KEY")
if not serper_api_key:
    print("Error: SERPER_API_KEY environment variable is not set.")
    sys.exit(1)

# Set up argument parser
parser = argparse.ArgumentParser(description="AI Research Agent System")
parser.add_argument("query", type=str, help="The query to search for.")
args = parser.parse_args()

# Create an LLM instance for Ollama LLaMA 3.2
llama_model = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"  # URL for Ollama's local server
)

# Initialize the agents
web_crawler_agent = WebCrawlerAgent(
    llm=llama_model,
    role="Web Crawler",
    backstory="I search the web to gather information on a given topic.",
    goal="Fetch web search results for a given query.",
    serper_api_key=serper_api_key
)

web_crawler_task = Task(
    description=args.query,  # Use the query passed from the command line
    expected_output="A detailed response about the latest advancements in AI research.",
    agent=web_crawler_agent
)

summarizer_agent = SummarizerAgent(llama_model)
summarizer_task = Task(description="Summarize the web data.", expected_output="Concise summaries of the search results.", agent=summarizer_agent)

report_generator_agent = ReportGeneratorAgent(llama_model)
report_task = Task(description="Generate a final research report.", expected_output="A comprehensive research report.", agent=report_generator_agent)

# Create the crew with all agents
crew = Crew(
    agents=[web_crawler_agent, summarizer_agent, report_generator_agent],
    tasks=[web_crawler_task, summarizer_task, report_task],
    verbose=True,
    process=Process.sequential  # Ensure tasks are executed in sequence
)

# Execute tasks in sequence
output = crew.kickoff()

# Print the final output
print(output)
