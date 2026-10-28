"""Render the same translation diagnostics for fresh and replayed chat answers."""
def render_translation_status(st, info, *, probe_key=None):
    if not info:
        st.caption('इस पुराने उत्तर का अनुवाद रिकॉर्ड उपलब्ध नहीं है। जाँच के लिए सवाल दोबारा पूछें।')
        return
    status = info.get('status')
    captions = {
        'not_configured': 'हिंदी अनुवाद चालू नहीं है: Sarvam की API कुंजी उपलब्ध नहीं है।',
        'disabled': 'हिंदी अनुवाद बंद है।',
        'fallback': 'हिंदी अनुवाद पूरा नहीं हुआ।',
    }
    reasons = {
        'http_400': 'अनुवाद सेवा ने अनुरोध स्वीकार नहीं किया।',
        'http_401': 'Sarvam API की कुंजी या अनुमति जाँचें।',
        'http_403': 'Sarvam API की कुंजी या अनुमति जाँचें।',
        'http_422': 'अनुवाद सेवा ने अनुरोध का प्रारूप स्वीकार नहीं किया।',
        'http_429': 'अनुवाद सेवा की अनुरोध सीमा पूरी हो गई है।',
        'TimeoutError': 'अनुवाद सेवा ने समय पर उत्तर नहीं दिया।',
        'request_limit': 'इस उत्तर में अनुवाद के लिए बहुत अधिक अलग-अलग अंश थे।',
        'translation_deadline': 'पूरे उत्तर के अनुवाद की समय सीमा समाप्त हुई।',
        'protected_values_changed': 'अनुवाद में सुरक्षित रखी गई मात्रा या नाम बदल गए; उत्तर स्वीकार नहीं हुआ।',
        'protected_fragment_limit': 'इस अंश में बहुत अधिक मात्रा या नाम थे; सुरक्षित अनुवाद पूरा नहीं हो सका।',
        'protected_values_lost': 'अनुवाद में मात्रा या नाम खो गए; उत्तर स्वीकार नहीं हुआ।',
        'untranslated_text_or_new_numbers': 'अनुवाद में अंग्रेज़ी पाठ या नई संख्याएँ मिलीं; उत्तर स्वीकार नहीं हुआ।',
        'untranslated_text': 'मान्य तकनीकी नामों के अलावा अंग्रेज़ी वाक्य या पाठ बचा है; अनुवाद पूरा नहीं हुआ।',
        'new_numbers': 'अनुवाद की संख्याएँ स्रोत की मात्राओं से मेल नहीं खातीं; उत्तर स्वीकार नहीं हुआ।',
        'hindi_missing': 'अनुवाद के अपेक्षित पाठ में हिंदी वाक्य नहीं मिले।',
        'invalid_translation_structure': 'अनुवाद सेवा ने सभी पाठ-अंश सही प्रारूप में नहीं लौटाए।',
        'empty_response': 'अनुवाद सेवा का उत्तर खाली था।',
    }
    if status in captions:
        st.caption(' '.join(filter(None, [captions[status], reasons.get(info.get('reason'))])))
    with st.expander('हिंदी अनुवाद की स्थिति'):
        # Safe status fields and words attested in the public guide; no provider bodies or credentials.
        st.json({k: info[k] for k in ('provider', 'model', 'retry_model', 'version', 'status', 'reason', 'request_chunks', 'failure_scope', 'source_fragment_kind', 'structure_error', 'remaining_latin_count', 'remaining_latin_words') if k in info})
        if status == 'fallback' and probe_key is not None:
            st.caption('यह स्थिर नमूने की सेवा-जाँच है, खेती की सलाह नहीं। नतीजे पाँच मिनट तक दोबारा उपयोग हो सकते हैं।')
            if st.button('Test Sarvam Hindi services', key='sarvam_probe_' + str(probe_key)) is True:
                from app.sarvam_diagnostics import run_diagnostics
                with st.spinner('Sarvam सेवा की जाँच हो रही है…'):
                    st.session_state['kisaanai_sarvam_probe'] = run_diagnostics()
            diagnostic = st.session_state.get('kisaanai_sarvam_probe')
            if isinstance(diagnostic, dict):
                st.json(diagnostic)
