import streamlit as st
from dotenv import load_dotenv
import PyPDF2
from elasticsearch import Elasticsearch, helpers
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from htmlTemplates import css, bot_template, user_template
import time
import requests
import json
from langchain.prompts import PromptTemplate

es_model_id = '.elser_model_2_linux-x86_64'
index_name = "elser_index_vb"
pipeline_name = "elser-ingest-pipeline"


def get_pdf_text(pdf_docs):
    text_content = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        text_content = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text_content


def get_text_chunks(text_content):
    # Split the text content into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_pages = splitter.split_text(text_content)  # Split the whole text_content

    bulk_data = []
    for index, text in enumerate(chunked_pages):
        doc = {
            'title': " page " + str(index + 1),
            'document': text
        }
        bulk_data += [doc]
    return bulk_data


def index_on_elastic_search(es_client, bulk_data):
    es_client.ingest.put_pipeline(id=pipeline_name, description="Ingest pipeline for ELSER",
                                  processors=[{"inference": {"model_id": es_model_id, "input_output": [
                                      {"input_field": "document", "output_field": "content_embedding"}]}}])
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name, ignore=[400, 404])
    es_client.indices.create(index=index_name, settings={"index": {"default_pipeline": pipeline_name}},
                             mappings={"properties": {"document": {"type": "text", "fields": {
                                 "keyword": {"type": "keyword", "ignore_above": 256}}},
                                                      "content_embedding": {"type": "sparse_vector"}}})
    res = helpers.bulk(es_client, bulk_data, index=index_name)
    print("indexing completed")
    time.sleep(5)
    return res


def query_elastic_search(es_client, user_question):
    print("Querying ELASTIC")

    # Elasticsearch query
    es_query = {
        "bool": {
            "should": [
                {"match": {"description": {"query": "", "boost": 1}}},
                {"text_expansion": {
                    "content_embedding": {"model_id": es_model_id, "model_text": user_question, "boost": 1}}}
            ]
        }
    }

    # Executing the search query
    es_response = es_client.search(index=index_name, size=5, query=es_query)
    title_docs = ""
    scoring_response_txt = dict(response="", references=[], text_max_overlap_score=0.0,
                                concatenated_text_overlap_score=0.0)

    # Processing search hits
    for hit in reversed(es_response.get('hits', {}).get('hits', [])):
        try:
            score = hit['_score']
            title = hit['_source']['title']
            document = hit['_source']['document']

            # Logging the hit details
            #print(f"\nRelevance Score  : {score}\nTitle            : {title}\nDocument Content : {document}\n")

            # Building response
            response_hit = {"document_title": title, "document_text": document, "hit_score": score}
            scoring_response_txt["references"].append(response_hit)

            # Appending document title and content
            title_docs += f"[title]: {title}\n[document]: {document}\n"


        except KeyError as e:
            print(f"KeyError: {e}")
        except Exception as e:
            print(f"Error in the response: {str(e)}")

    return title_docs


def LLM_Response(title_docs, user_question):
    BAM_API_KEY = os.environ.get('BAM_API_KEY')
    BAM_url = os.environ.get('BAM_URL')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + BAM_API_KEY
    }
    prompt_text = "[INST]<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Be brief in your answers. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'\''t know the answer to a question, please do not share false information. <</SYS>>\nGenerate the next agent response by answering the question. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities and use the tiles of documents to separate between topics or domains. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer.\n<title_docs><</SYS>>\n\n<question> Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer. [/INST]"
    prompt_text = prompt_text.replace('<title_docs>', title_docs).replace('<question>', user_question)
    print("printing prompt text")
    print(prompt_text)
    print("Ending prompt text")
    data = {
        "model_id": "meta-llama/llama-2-13b-chat",
        "input": prompt_text,
        "parameters": {
            "decoding_method": "greedy",
            "min_new_tokens": 1,
            "max_new_tokens": 500
        }
    }

    response = requests.post(BAM_url, headers=headers, data=json.dumps(data))
    return json.loads(response.text)["results"][0]['generated_text']


def handle_userinput(es_client,user_question):

    es_response = query_elastic_search(es_client, user_question)
    llm_response = LLM_Response(es_response, user_question)
    add_to_chat(user_question, llm_response)

    for role, message in st.session_state.chat_history:
        if role == "User":
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


def add_to_chat(user_input, bot_response):
    #st.session_state.chat_history.insert(0, ("User", user_input))
    #st.session_state.chat_history.insert(0, ("Bot", bot_response))

    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))

def main():
    load_dotenv()

    es_client = Elasticsearch(
        os.environ["elastic_search_url"],
        api_key=os.environ["elastic_search_api_key"]
    )

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(es_client, user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                bulk_data = get_text_chunks(raw_text)


                index_on_elastic_search(es_client, bulk_data)
                # st.session_state.conversation = query_elastic_search(es_client, "vikram bhat")


if __name__ == '__main__':
    main()
