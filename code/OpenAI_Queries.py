from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import traceback
from utilities.helper import LLMHelper
import time

import logging
logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

def check_deployment():
    # Check if the deployment is working
    #\ 1. Check if the embedding is working
    try:
        llm_helper = LLMHelper()
        llm_helper.embeddings.embed_documents(texts=["This is a test"])
        st.success("Embedding is working!")
    except Exception as e:
        st.error(f"""Embedding model is not working.  
            Please check you have a deployment named "text-embedding-ada-002" for "text-embedding-ada-002" model in your Azure OpenAI resource {llm_helper.api_base}.  
            Then restart your application.
            """)
        st.error(traceback.format_exc())

    #\ 4. Check if the Redis is working with previous version of data
    try:
        llm_helper = LLMHelper()
        if llm_helper.vector_store.check_existing_index("embeddings-index"):
            st.warning("""Seems like you're using a Redis with an old data structure.  
            If you want to use the new data structure, you can start using the app and go to "Add Document" -> "Add documents in Batch" and click on "Convert all files and add embeddings" to reprocess your documents.  
            To remove this working, please delete the index "embeddings-index" from your Redis.  
            If you prefer to use the old data structure, please change your Web App container image to point to the docker image: fruocco/oai-embeddings:2023-03-27_25. 
            """)
        else:
            st.success("Redis is working!")
    except Exception as e:
        st.error(f"""Redis is not working. 
            Please check your Redis connection string in the App Settings.  
            Then restart your application.
            """)
        st.error(traceback.format_exc())


def calcola_categoria():
    try:
        print('calcola categoria')
    except Exception as e:
        st.error(traceback.format_exc())

try:
    st.title("OpenAI - Categorizzazione Mail")
    
    llm_helper = LLMHelper()

    st.button("Check Ambiente", on_click=check_deployment)

    mail = st.text_area("Prova Categorizzazione Mail:", height=300, placeholder="testo della mail da categorizzare")

    if st.button("Calcola", on_click=calcola_categoria):
        st.session_state['question'], st.session_state['response'], st.session_state['context'], sources = llm_helper.get_categoization(mail, [])
        st.markdown("Answer:" + st.session_state['response'])
        st.markdown(f'\n\nSources: {sources}') 
        with st.expander("Question and Answer Context"):
            st.markdown(st.session_state['context'].replace('$', '\$'))
            st.markdown(f"SOURCES: {sources}") 

except Exception:
    st.error(traceback.format_exc())
