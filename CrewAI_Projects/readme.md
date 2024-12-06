
# Query-Based Research Report Generator

This project is a Query-Based Research Report Generator powered by LLaMA 3.2 and Crewa AI agents. It allows users to enter a query, fetch information from the web, summarize the data, and generate a detailed report. The system can generate reports in multiple styles such as "Detailed Analysis", "Brief Summary", and "Bullet Points". The project leverages Gradio for creating an interactive web interface.

## Features

- **Web Crawling**: Fetches relevant information based on the user's query using the web crawler agent.
- **Summarization**: Condenses the fetched data into concise summaries.
- **Report Generation**: Generates a comprehensive research report in the selected style (e.g., Detailed Analysis, Brief Summary, Bullet Points).
- **Interactive Interface**: Gradio interface for easy interaction with the system.
- **Customizable Report Styles**: Select between various report styles based on the query.

## Project Structure

The project is organized into the following structure:

```
CrewAI_Projects/
├── agents/
│   ├── web_crawler_agent.py
│   ├── summarizer_agent.py
│   └── report_generator_agent.py
├── data/
│   ├── data_file1.csv
│   └── data_file2.json
├── main.py
├── .env
├── requirements.txt
└── README.md
```

### Description of the folders and files:
- **`agents/`**: Contains Python scripts for different AI agents such as `web_crawler_agent.py`, `summarizer_agent.py`, and `report_generator_agent.py`.
- **`main.py`**: The main script that runs the app and integrates all the tasks performed by the agents.
- **`.env`**: Stores environment variables such as API keys securely.
- **`requirements.txt`**: Contains a list of dependencies required to run the project.
- **`README.md`**: Provides an overview and instructions for the project.

## Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.x
- Ollama LLaMA 3.2 (installed locally)
- Crewa AI SDK
- Other dependencies listed in the `requirements.txt` file.

### Environment Setup

Create a `.env` file in the root directory with the following content:

```env
SERPER_API_KEY=<your-serper-api-key>
```

Replace `<your-serper-api-key>` with your actual API key for the SerpAPI.

## Installation

1. Clone this repository:


2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Launch the app:
    ```
    python main.py
    ```

2. Open the Gradio interface in your browser. Enter your query, select a report style, and click "Generate Report" to view the output.

3. The generated report will appear on the interface, formatted according to the selected style.

## Agents and Task Flow

This project uses three AI agents to perform different tasks:
- **WebCrawlerAgent**: Fetches relevant data from the web.
- **SummarizerAgent**: Summarizes the fetched data.
- **ReportGeneratorAgent**: Compiles the final research report with any bias removed.

These agents are orchestrated using the `Crew` class from the Crewa AI SDK, and tasks are executed in sequence for optimal performance.

## Contributing

We welcome contributions to improve the functionality and features of this project. If you'd like to contribute, follow these steps:
1. Fork this repository.
2. Create a new branch for your changes.
3. Make your changes and submit a pull request.


## Acknowledgments

- **Crew AI SDK**: For providing the framework to build multi-agent systems.
- **Ollama LLaMA 3.2**: For powering the language model used in this project.
- **SerpAPI**: For enabling web crawling capabilities.
- **Gradio**: For the easy-to-use UI framework.

```
