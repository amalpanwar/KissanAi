"""Native Streamlit composer with editable suggested questions."""
from __future__ import annotations

SUGGESTIONS = (
    ("🌤️ मौसम", "आज मेरे क्षेत्र में मौसम कैसा रहेगा?"),
    ("💰 मंडी भाव", "आज गेहूं का मंडी भाव क्या है?"),
    ("🌱 फसल उगाना", "गेहूं की खेती कैसे करें?"),
    ("🐛 कीटनाशक", "गेहूं में दीमक की रोकथाम के लिए कौन सा कीटनाशक उपयोग करें?"),
)
CHAT_KEY = "farmer_chat_input"
DRAFT_KEY = "suggested_question_draft"


def render_chat_composer(st):
    """Select/edit a draft; submit through the same return path as typed chat."""
    st.caption("सुझाया सवाल चुनें, जरूरत हो तो फसल बदलें, फिर भेजें। मौसम और भाव के लिए चुना हुआ स्थान इस्तेमाल होगा।")

    def choose(question):
        st.session_state[DRAFT_KEY] = question

    def discard():
        st.session_state.pop(DRAFT_KEY, None)

    def submit():
        draft = st.session_state.get(DRAFT_KEY, "").strip()
        if draft:
            st.session_state["submitted_suggestion"] = draft
            discard()

    for column, (label, question) in zip(st.columns(4), SUGGESTIONS):
        column.button(label, key=f"suggest_{label}", help=question,
                      on_click=choose, args=(question,), use_container_width=True)
    if st.session_state.get(DRAFT_KEY):
        with st.form("suggested_question_form"):
            st.text_input("सवाल बदलें या भेजें", key=DRAFT_KEY)
            st.form_submit_button("सवाल भेजें", on_click=submit)
            st.form_submit_button("रद्द करें", on_click=discard)
    typed = st.chat_input("अपना सवाल लिखें...", key=CHAT_KEY, on_submit=discard)
    selected = st.session_state.pop("submitted_suggestion", None)
    return typed or selected
