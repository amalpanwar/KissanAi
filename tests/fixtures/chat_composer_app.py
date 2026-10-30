from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st
from app.chat_controls import render_chat_composer
from app.chat_session import sync_chat_location

st.title('Chat composer integration test')
if st.button('Rerun unrelated control'):
    pass
place = st.selectbox('Selected village', ['Doghat Rural', 'Another village'])
sync_chat_location(st.session_state, {'state': 'Uttar Pradesh', 'district': 'Baghpat', 'sub_district': 'Baraut', 'place': place})
response_area = st.container()
question = render_chat_composer(st)
if question:
    st.session_state.setdefault('chat_history', []).append(question)
with response_area:
    for index, text in enumerate(st.session_state.get('chat_history', [])):
        with st.chat_message('user'):
            st.write(text)
        with st.chat_message('assistant'):
            st.write(f'Generated response {index + 1}: {text}')
    st.json(st.session_state.get('chat_history', []))
