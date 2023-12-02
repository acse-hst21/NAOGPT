import streamlit as st
import torch
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from HTML_Templates import css, user_template, bot_template

def set_device() -> str:
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def get_pdf_text(pdf_files: list) -> str:
    text = ''
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text: str) -> list:
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    return chunks


def get_vectorstore(text_chunks: list) -> FAISS:
    embeddings = HuggingFaceInstructEmbeddings(model_name=
                                               'hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, 
                                   embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                         model_kwargs={"temperature":0.5,
                                       "max_length":64})
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def answer_question(user_question: str) -> None:
    answer = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = answer['chat_history']

    for index, message in enumerate(st.session_state.chat_history):
        if index % 2 == 0:
            st.write(user_template.replace('{{MSG}}', message.content),
            unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}', message.content),
            unsafe_allow_html=True)


def main() -> None:
    load_dotenv()
    # device = set_device()

    st.set_page_config(page_title='NAOGPT', page_icon=':books:')
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header('NAOGPT')
    user_question = st.text_input('Ask me a question:')
    if user_question:
        answer_question(user_question)

    with st.sidebar:
        st.subheader('Your documents')
        pdf_files = st.file_uploader(
            'Upload your PDFs here and click on "Process"',
            accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Loading...'):
                raw_text = get_pdf_text(pdf_files)
                
                text_chunks = get_text_chunks(raw_text)
                
                vector_store = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vector_store)
            st.write('Process complete!')
                
if __name__ == '__main__':
    main()