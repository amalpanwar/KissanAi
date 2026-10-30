"""Render the same translation diagnostics for fresh and replayed chat answers."""
def render_translation_status(st, info):
    if not info:
        st.caption('इस पुराने उत्तर का अनुवाद रिकॉर्ड उपलब्ध नहीं है। जाँच के लिए सवाल दोबारा पूछें।')
        return
    status = info.get('status')
    captions = {
        'not_configured': 'हिंदी अनुवाद चालू नहीं है: Sarvam की API कुंजी उपलब्ध नहीं है।',
        'disabled': 'हिंदी अनुवाद बंद है। मूल उत्तर दिखाया गया है।',
        'fallback': 'हिंदी अनुवाद पूरा नहीं हुआ; मूल उत्तर दिखाया गया है।',
    }
    if status in captions:
        st.caption(captions[status])
    with st.expander('हिंदी अनुवाद की स्थिति'):
        # Show only non-sensitive, stable fields, never service bodies or credentials.
        st.json({k: info[k] for k in ('provider', 'model', 'version', 'status', 'reason') if k in info})
