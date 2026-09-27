"""Native Streamlit composer with editable suggested questions."""
from __future__ import annotations

SUGGESTIONS = (
    ("🌤️ मौसम", "आज मेरे क्षेत्र में मौसम कैसा रहेगा?"),
    ("💰 मंडी भाव", "आज गेहूं का मंडी भाव क्या है?"),
    ("🌱 फसल उगाना", "गेहूं की खेती कैसे करें?"),
    ("🐛 कीटनाशक", "गेहूं में दीमक की रोकथाम के लिए कौन सा कीटनाशक उपयोग करें?"),
)
CHAT_KEY = "farmer_chat_input"


def render_chat_composer(st):
    """Fill the composer on selection; only Enter/Send submits the question."""
    if st.session_state.pop("clear_question_draft", False):
        st.session_state.pop("suggested_question_draft", None)
    st.caption("सुझाया सवाल चुनें, जरूरत हो तो फसल बदलें, फिर भेजें। मौसम और भाव के लिए चुना हुआ स्थान इस्तेमाल होगा।")

    def choose(question):
        st.session_state["suggested_question_draft"] = question

    for column, (label, question) in zip(st.columns(4), SUGGESTIONS):
        column.button(label, key=f"suggest_{label}", help=question,
                      on_click=choose, args=(question,), use_container_width=True)
    selected = None
    if st.session_state.get("suggested_question_draft"):
        with st.form("suggested_question_form"):
            draft = st.text_input("सवाल बदलें या भेजें", key="suggested_question_draft")
            send = st.form_submit_button("सवाल भेजें")
            cancel = st.form_submit_button("रद्द करें")
        if send and draft.strip():
            selected = draft.strip()
            st.session_state["clear_question_draft"] = True
        elif cancel:
            st.session_state["clear_question_draft"] = True
            st.rerun()
    typed = st.chat_input("अपना सवाल लिखें...", key=CHAT_KEY)
    return typed or selected
