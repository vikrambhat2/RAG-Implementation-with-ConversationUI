from crewai import Agent,  Task
import requests

class WebCrawlerAgent(Agent):
    def __init__(self, llm, role, backstory, goal, serper_api_key):
        super().__init__(llm=llm, role=role, backstory=backstory, goal=goal)
        self._serper_api_key = serper_api_key

    def web_crawl(self, query):
        """Fetch search results from the Serper.dev API."""
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self._serper_api_key}
        payload = {"q": query}
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            try:
                results = response.json()
                return results.get("organic", [])
            except Exception as e:
                raise ValueError(f"Failed to parse JSON: {e}")
        else:
            raise Exception(f"Serper API error: {response.status_code}, {response.text}")

    def execute_task(self, task: Task, context: dict = None, tools: list = None):
        """Execute the task by performing a web search and returning results as a string."""
        query = task.description
        if not query:
            raise ValueError("Task description must include a 'query' field.")
        
        # Perform the web crawl
        search_results = self.web_crawl(query)
        
        # Format the search results as a string (e.g., by joining titles)
        search_results_str = "\n".join(
            [result.get('title', 'No Title') for result in search_results]
        )
        
        # Ensure we return a string as expected by TaskOutput
        return search_results_str
