# Streamlit Reddit App

This is a Streamlit application designed to summarize lengthy Reddit posts or discussions and answer questions based on the content. It leverages the Reddit API for fetching posts and subreddits, and utilizes a Retrieval-Augmented Generation (RAG) system to process the data and provide relevant answers.

## Features

- **Reddit Post Summarization**: Fetch and summarize long Reddit threads or posts.
- **Question Answering**: Ask questions related to specific Reddit discussions, and get relevant answers based on the post content.
- **RAG Integration**: Uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant information and generate answers.
- **User-friendly Interface**: A simple and interactive interface built with Streamlit for seamless user experience.

## Requirements

To run this app, you need the Python libraries specified in requirements.txt file:

### Running the App:

Ensure you have your environment variables set up for Reddit API and the LLM API.
Run the app with the following command:
```
streamlit run Streamlit_redditApp.py
```

