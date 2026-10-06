"""Sidebar news view; shares the same cached source as the news agent."""
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
from app.agriculture_news import SOURCE_URL, get_news


@st.fragment(run_every=900)
def render_news_panel():
    st.subheader('🌾 कृषि समाचार')
    st.caption('Down To Earth Hindi · राष्ट्रीय और अंतरराष्ट्रीय खबरें')
    snapshot = get_news()
    if snapshot['status'] == 'unavailable':
        st.info('समाचार अभी उपलब्ध नहीं हैं। अगली बार फिर प्रयास करेंगे।')
    else:
        if snapshot['status'] == 'stale':
            st.warning('लाइव अपडेट उपलब्ध नहीं; पिछली सुरक्षित सूची।')
        for item in snapshot['articles'][:5]:
            st.link_button(item['title'], item['url'], use_container_width=True)
            published = datetime.fromisoformat(item['published_at']).astimezone(ZoneInfo('Asia/Kolkata'))
            st.caption(f"प्रकाशित: {published:%d-%m-%Y %H:%M} IST")
        checked = datetime.fromtimestamp(snapshot['fetched_at'], ZoneInfo('Asia/Kolkata'))
        st.caption(f'सूची जाँची गई: {checked:%d-%m-%Y %H:%M} IST · हर 15 मिनट अपडेट')
    st.link_button('सभी कृषि समाचार पढ़ें', SOURCE_URL)
    st.caption('चैट में पूछें: “आज की कृषि खबरें” या “प्याज की ताज़ा खबर”।')
