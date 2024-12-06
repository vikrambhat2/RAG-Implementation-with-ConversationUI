from crewai import Agent, Task

class SummarizerAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="Content Summarizer",
            backstory="I summarize research content into concise insights.",
            goal="Summarize research articles or content to provide clear and precise information."
        )

            

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        """Execute the summarization task."""
        if not context:
            raise ValueError("Summarization task requires content in context.")
        
        # Raw content might be a string or a list of strings
        raw_content = context
        if isinstance(raw_content, str):
            summary = self.summarize(raw_content)
        elif isinstance(raw_content, list) and all(isinstance(item, str) for item in raw_content):
            summary = self.summarize("\n".join(raw_content))  # Combine all strings into one for summarization
        else:
            raise ValueError("Raw content must be a string or a list of strings.")
        
        return summary

    def summarize(self, content):
        """Summarize the content using the LLM."""
        # Directly summarize the content using LLM's capabilities
        #return self.llm(content)
        return self.llm.call([
            {"role": "user", "content": content}
        ])