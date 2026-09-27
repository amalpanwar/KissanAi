from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from app.chat_controls import render_chat_composer

st.title('Chat composer integration test')
if st.button('Rerun unrelated control'):
    pass
question = render_chat_composer(st)
if question:
    st.session_state.setdefault('submitted', []).append(question)
st.json(st.session_state.get('submitted', []))
