import streamlit as st
import os
import traceback
from utilities.helper import LLMHelper

try:

    llm_helper = LLMHelper()

    # Query RediSearch to get all the embeddings
    data = llm_helper.get_all_documents(k=1000)

    if len(data) == 0:
        st.warning("No embeddings found.")
    else:
        st.dataframe(data, use_container_width=True)
        
        st.download_button("Download data", data.to_csv(index=False).encode('utf-8'), "embeddings.csv", "text/csv", key='download-embeddings')

        st.text("")
        st.text("")
        col1, col2, col3, col4 = st.columns([3,2,2,1])
        with col1:
            st.selectbox("Embedding id to delete", data.get('key',[]), key="embedding_to_drop")
        with col2:
            st.text("")
            st.text("")
            st.button("Delete embedding", on_click=delete_embedding)
        with col3:
            st.selectbox("File name to delete", set(data.get('filename',[])), key="file_to_drop")
        with col4:
            st.text("")
            st.text("")
            st.button("Delete file", on_click=delete_file)

        st.text("")
        st.text("")
        st.button("Delete all embeddings", on_click=llm_helper.vector_store.delete_keys_pattern, args=("doc*",), type="secondary")


except Exception as e:
    st.error(traceback.format_exc())
