import streamlit as st
from dotenv import load_dotenv
import os
from htmlTemplate import css, bot_template, user_template
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import PyPDF2
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_community.llms import WatsonxLLM
from elasticsearch import Elasticsearch
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util

load_dotenv()
es_model_id = '.elser_model_2_linux-x86_64'
index_name = "elser_index_vb_12"
llm_model_id = "meta-llama/llama-2-13b-chat"
wx_url = "https://us-south.ml.cloud.ibm.com"
wx_project_id = "b33db82c-437e-4d87-8b9c-719e9919003e"


def prepare_docs(pdf_docs):
    docs = []
    metadata = []
    content = []

    for pdf in pdf_docs:
        print(pdf.name)
        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf.name + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    return content, metadata


def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Split documents into {len(split_docs)} passages")
    return split_docs


def ingest_and_get_vector_store(split_docs):
    vector_store = ElasticsearchStore(
        es_url=os.environ["elastic_search_url"],
        es_api_key=os.environ["elastic_search_api_key"],
        index_name=index_name,
        strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=es_model_id)
    )
    documents = vector_store.from_documents(
        split_docs,
        es_url=os.environ["elastic_search_url"],
        es_api_key=os.environ["elastic_search_api_key"],
        index_name=index_name,
        strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=es_model_id)
    )

    return vector_store


def get_conversation_chain(vector_store):
    parameters = {
        GenParams.DECODING_METHOD: "sample",
        GenParams.MAX_NEW_TOKENS: 100,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.TEMPERATURE: 0.5,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1,
    }

    watsonx_llm = WatsonxLLM(
        model_id=llm_model_id,
        url=wx_url,
        project_id=wx_project_id,
        params=parameters,
        apikey=os.environ["WATSONX_APIKEY"]
    )
    retriever = vector_store.as_retriever()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Be brief in your answers. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'''t know the answer to a question, please do not share false information.
    Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = (ConversationalRetrievalChain.from_llm
                          (llm=watsonx_llm,
                           retriever=retriever,
                           # condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=True))
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



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
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                content, metadata = prepare_docs(pdf_docs)

                # get the text chunks
                split_docs = get_text_chunks(content, metadata)

                # create vector store
                vectorstore = ingest_and_get_vector_store(split_docs)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
