from crewai import Agent, Task

class ReportGeneratorAgent(Agent):
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            role="report generator",
            backstory="I generate a final comprehensive report without a bias",
            goal="Create a detailed report summarizing the research findings."
        )

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        """Generate a comprehensive report."""
        
        summaries =  context #task.description  # Get the summaries and bias analysis
        report = self.generate_report(summaries)
        return report

    def generate_report(self, summaries):
        """Generate a comprehensive research report."""
        #return self.llm.execute_task(Task(description=f"Generate a report from: {summaries}"))
        return self.llm.call([
                        {"role": "user", "content": summaries}
                    ])
