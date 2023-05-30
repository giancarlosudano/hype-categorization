import streamlit as st
import os, json, re, io
from os import path
import requests
import mimetypes
import traceback
import chardet
from utilities.helper import LLMHelper
import uuid
from redis.exceptions import ResponseError 
from urllib import parse
import time
import hashlib

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from utilities.formrecognizer import AzureFormRecognizerClient
from utilities.azureblobstorage import AzureBlobStorageClient
from utilities.translator import AzureTranslatorClient
from utilities.customprompt import PROMPT
from utilities.redis import RedisExtended

def upload_all_files():
    local_path = st.session_state['local_path']
    count = os.listdir(local_path).count
    st.text(f"File recuperati : {count}")

    for file_name in os.listdir(local_path):
        try:
            with open(os.path.join(local_path, file_name), "r", encoding="utf-8") as file:
                testo = file.read()
                st.text(file_name)    
                st.text("Upload Start...")
                source_url = llm_helper.blob_client.upload_file(testo, file_name=file_name, content_type='text/plain; charset=utf-8')
                st.text("Sleep time")          
                time.sleep(st.session_state['delay'])
                st.text("Start Embedding...")
                llm_helper.add_embeddings_lc(source_url)
                st.success(f"Embedded {file_name}")
                st.text("Sleep time")          
                time.sleep(st.session_state['delay'])
        except Exception as e:
            st.error(traceback.format_exc())
    
    st.text("Fine elaborazione")    

try:
    st.title("OpenAI - Embedding Files")
    llm_helper = LLMHelper()
    st.session_state['local_path'] = st.text_input("Percorso locale dei file da embeddare:", value="C:\\NoSync\\Hype\\splitted")
    st.session_state['delay'] = st.slider("Delay tra un file e l'altro:", 1, 90, 1)
    st.button("Compute Embeddings" , on_click=upload_all_files)

except Exception as e:
    st.error(traceback.format_exc())
