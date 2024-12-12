
# YouTube Video Q&A System Using RAG, LangChain, Ollama Llama 3.3, and Gradio UI

This project enables users to extract and summarize content from YouTube videos and interact with the content through a Q&A system. By leveraging an advanced Retrieval-Augmented Generation (RAG) approach, LangChain, Ollama Llama 3.3, and Gradio UI, this system allows users to efficiently query and retrieve contextually relevant answers based on YouTube video captions (or transcripts).

## Features

- **YouTube Video Summarization**: Summarize the content of any YouTube video by providing its URL.
- **Interactive Q&A**: Ask questions about the content of the video, and get precise, context-aware answers in real-time.
- **Seamless User Interface**: Built with Gradio for an easy-to-use interface where users can submit YouTube URLs and ask questions interactively.
- **Efficient Video Processing**: Uses LangChain to process video captions, split the transcript into manageable chunks, and store them in a FAISS vector database for efficient querying.
  
## Prerequisites

To run this project, you need to have the following Python libraries installed:

- **gradio**: For creating the interactive user interface.
- **langchain**: A framework for building applications that integrate with large language models (LLMs), enabling composability and easy creation of complex workflows.
- **langchain_huggingface**: Integration with HuggingFace models for embeddings.
- **langchain_community**: Community extensions for LangChain.
- **langchain_ollama**: Integration with Ollama Llama 3.3 for conversational capabilities.
- **faiss-cpu**: FAISS library for efficient similarity search.

Install the required dependencies using:

```
pip install -r requirements.txt
```

## Setup Instructions

1. Clone the repository:



2. Install the required dependencies (as mentioned above).

3. Run the Gradio interface:

   ```
   python converse_with_yt_videos.py
   ```

4. The interface will launch in your browser where you can:
   - Enter a YouTube video URL to index the video.
   - Ask questions about the video, and receive relevant answers.

## How It Works

1. **YouTube Video URL**: The system takes a YouTube URL as input and loads the video's captions or transcripts (in English by default). 
2. **Text Processing**: The transcript is split into manageable text chunks for easier processing.
3. **Embeddings**: Text chunks are converted into vector embeddings using a HuggingFace model.
4. **Vector Database**: These embeddings are stored in a FAISS vector database, enabling fast similarity searches.
5. **Summarization & Q&A**: Users can ask questions, and the system will retrieve the most relevant portions of the transcript to provide precise answers.

### Modify Language Preferences
By default, the system processes English captions. If you want to modify the language preferences for captions, refer to [LangChain's YouTube Transcript loader documentation](https://python.langchain.com/docs/integrations/document_loaders/youtube_transcript/#add-language-preferences).

## Example

### Summarize a Video
1. Enter the YouTube URL in the textbox.
2. Click "Submit URL."
3. The system will display a summary of the video.

### Ask a Question
1. After the video is indexed, enter a question related to the video content.
2. Click "Ask Question."
3. The system will return a contextually relevant answer based on the video transcript.

## Conclusion

This system leverages cutting-edge AI technologies to streamline the way we consume YouTube videos. It saves time by summarizing content and answering questions interactively, allowing users to quickly extract the information they need without manually searching through long videos.
