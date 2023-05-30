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
    
def upload_text_and_embeddings():
    file_name = f"{uuid.uuid4()}.txt"
    source_url = llm_helper.blob_client.upload_file(st.session_state['doc_text'], file_name=file_name, content_type='text/plain; charset=utf-8')
    llm_helper.add_embeddings_lc(source_url) 
    st.success("Embeddings added successfully.")

try:
    st.title("OpenAI - Embedding Sample Text")
    llm_helper = LLMHelper()

    st.session_state['doc_text'] = st.text_area("Nuovi esempi di categorizzazione", placeholder=
        """Aggiungere esempi di categorizzazione nel formato:
        ###MAIL
        Contenuto dellal mail

        TAG: uno dei tag predefiniti
        """, height=500)
    
    if st.button("Compute Embeddings", on_click=upload_text_and_embeddings):
        source_url = llm_helper.blob_client.upload_file(st.session_state['doc_text'], file_name="filename001.txt", content_type='text/plain; charset=utf-8')
        llm_helper.add_embeddings_lc(source_url)
        st.success("Embeddings added successfully.")
    
except Exception as e:
    st.error(traceback.format_exc())