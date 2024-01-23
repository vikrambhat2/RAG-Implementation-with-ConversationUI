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
index_name = "elser_index_vb_test_1"
llm_model_id = "meta-llama/llama-2-13b-chat"
wx_url = os.environ["WATSONX_URL"]
wx_project_id = os.environ["WATSONX_Project_ID"]  

llmtemplate = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}
[/INST]
"""

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
                    es_cloud_id= os.environ["elastic_search_cloud_id"],
                    es_api_key=os.environ["elastic_search_api_key"],
                    index_name=index_name,
                    strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=es_model_id)
                    )
    documents = vector_store.from_documents(
        split_docs,
        es_cloud_id= os.environ["elastic_search_cloud_id"],
        es_api_key=os.environ["elastic_search_api_key"],
        index_name=index_name,
        strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(model_id=es_model_id)
    )

    return vector_store


def get_conversation_chain(vector_store):
    parameters = {
        GenParams.DECODING_METHOD: "sample",
        GenParams.MAX_NEW_TOKENS: 150,
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
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llmtemplate)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=watsonx_llm,
                           retriever=retriever,
                           #condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=True)
    return conversation_chain

def validate_answer_against_sources(response_answer, source_documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarity_threshold = 0.3  
    source_texts = [doc.page_content for doc in source_documents]

    answer_embedding = model.encode(response_answer, convert_to_tensor=True)
    source_embeddings = model.encode(source_texts, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)
    print(cosine_scores)

    if any(score.item() > similarity_threshold for score in cosine_scores[0]):
        return True  

    return False  


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    response_answer = response['answer']

    print(response['answer'])
    
    is_valid_answer = validate_answer_against_sources(response['answer'], response['source_documents'])
    if not is_valid_answer or not response['source_documents']:
        response_answer = "Sorry I can not answer the question based on the given documents"



    add_to_chat(user_question, response_answer )

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
