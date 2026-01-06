import streamlit as st

# 1. AYAR KOMUTU BURADA OLMAK ZORUNDA (Importlardan Önce!)
st.set_page_config(
    page_title="Fetal Sağlık AI Projesi", 
    layout="wide", 
    page_icon="🧬"
)

# 2. Diğer importlar ayardan SONRA gelmeli
from ui.interface import run_ui 

if __name__ == "__main__":
    run_ui()