# from dotenv import load_dotenv
# load_dotenv()
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import streamlit as st
import pathlib
import tempfile
import os


#제목
st.title("ChatCSV")
st.write("---")

#OpenAI KEY 입력 받기
# openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

#파일 업로드
uploaded_file = st.file_uploader("PDF, TXT, CSV 파일을 올려주세요!",type=['pdf', 'txt', 'csv'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:

   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="euc-kr", csv_args={
                'delimiter': ','})
    data = loader.load()
  
    # st.write(data)

    #Split
    # text_splitter = RecursiveCharacterTextSplitter(
    #     # Set a really small chunk size, just to show.
    #     chunk_size = 300,
    #     chunk_overlap  = 20,
    #     length_function = len,
    #     is_separator_regex = False,
    # )
    # texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()
    vectorstores = FAISS.from_documents(data, embeddings_model)
    #persist_directory
    # persist_directory="C:\langchain/chatpdf2/"

    # load it into Chroma
    # db = Chroma.from_documents(texts, embeddings_model)
    # db = Chroma.from_documents(data, embeddings_model, persist_directory=persist_directory)
    # db.persist()

    #Question
    st.header("CSV에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            # chain = ConversationalRetrievalChain.from_llm(
            #     llm = ChatOpenAI(tempfile=0.0, model_name='gpt-3.5-turbo'),
            #     retriever=vectorstores.as_retriever
            # )
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstores.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])