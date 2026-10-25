"""Render the same translation diagnostics for fresh and replayed chat answers."""
def render_translation_status(st, info):
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
        'protected_values_lost': 'अनुवाद में मात्रा या नाम खो गए; उत्तर स्वीकार नहीं हुआ।',
        'untranslated_text_or_new_numbers': 'अनुवाद में अंग्रेज़ी पाठ या नई संख्याएँ मिलीं; उत्तर स्वीकार नहीं हुआ।',
        'hindi_missing': 'अनुवाद सेवा ने हिंदी उत्तर नहीं लौटाया।',
        'empty_response': 'अनुवाद सेवा का उत्तर खाली था।',
    }
    if status in captions:
        st.caption(' '.join(filter(None, [captions[status], reasons.get(info.get('reason'))])))
    with st.expander('हिंदी अनुवाद की स्थिति'):
        # Show only non-sensitive, stable fields, never service bodies or credentials.
        st.json({k: info[k] for k in ('provider', 'model', 'version', 'status', 'reason') if k in info})

