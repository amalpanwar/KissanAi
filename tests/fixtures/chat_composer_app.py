from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from app.chat_controls import render_chat_composer

st.title('Chat composer integration test')
if st.button('Rerun unrelated control'):
    pass
response_area = st.container()
question = render_chat_composer(st)
if question:
    st.session_state.setdefault('submitted', []).append(question)
with response_area:
    for index, text in enumerate(st.session_state.get('submitted', [])):
        with st.chat_message('user'):
            st.write(text)
        with st.chat_message('assistant'):
            st.write(f'Generated response {index + 1}: {text}')
    st.json(st.session_state.get('submitted', []))
