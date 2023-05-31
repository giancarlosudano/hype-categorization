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

try:
    st.title("OpenAI - Embedding Files")
    llm_helper = LLMHelper()
    st.session_state['local_path'] = st.text_input("Percorso locale dei file da embeddare:", value="C:\\NoSync\\Hype\\splitted")
    if st.button("Compute Embeddings"):
        # local_path = st.session_state['local_path']
        # llm_helper.add_mail_embeddings(st.session_state['local_path'])
        llm_helper.test_vector_store()

except Exception as e:
    st.error(traceback.format_exc())
