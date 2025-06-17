import streamlit as st


def configure_streamlit():
    """Configure Streamlit for large file uploads"""
    # Set max upload size to 2GB (2048MB)
    st.set_page_config(page_title="🍕 Pizza Sales - Robust Tracking", page_icon="🍕", layout="wide")

    # Override default file upload limit
    st._config.set_option("server.maxUploadSize", 2048)  # 2GB
    st._config.set_option("server.maxMessageSize", 2048)  # 2GB
