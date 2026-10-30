"""Render local Hindi translation status for new and replayed chat answers."""


def render_translation_status(st, info, *, probe_key=None):
    if not info:
        st.caption('इस पुराने उत्तर का अनुवाद रिकॉर्ड उपलब्ध नहीं है। जाँच के लिए सवाल दोबारा पूछें।')
        return
    status = info.get('status')
    captions = {
        'not_configured': 'स्थानीय हिंदी अनुवाद मॉडल उपलब्ध नहीं है; स्रोत-आधारित अंग्रेज़ी उत्तर दिखाया गया है।',
        'disabled': 'हिंदी अनुवाद बंद है।',
        'fallback': 'हिंदी अनुवाद पूरा नहीं हुआ; स्रोत-आधारित अंग्रेज़ी उत्तर दिखाया गया है।',
        'partial': 'उत्तर आंशिक रूप से हिंदी में है; अनुवाद न हुए अंश अंग्रेज़ी में रखे गए हैं।',
    }
    reasons = {
        'TimeoutError': 'स्थानीय अनुवाद में समय लग गया।',
        'request_limit': 'इस उत्तर में अनुवाद के लिए बहुत अधिक अलग-अलग अंश थे।',
        'translation_deadline': 'पूरे उत्तर के अनुवाद की समय सीमा समाप्त हुई।',
        'protected_values_changed': 'अनुवाद में सुरक्षित मात्रा या नाम बदल गए; वह अंश अंग्रेज़ी में रखा गया।',
        'protected_values_lost': 'अनुवाद में मात्रा या नाम खो गए; वह अंश अंग्रेज़ी में रखा गया।',
        'untranslated_text': 'तकनीकी नामों के अलावा अंग्रेज़ी पाठ बचा है; वह अंश अंग्रेज़ी में रखा गया।',
        'new_numbers': 'अनुवाद में स्रोत से अलग संख्या मिली; वह अंश अंग्रेज़ी में रखा गया।',
        'hindi_missing': 'अनुवाद में हिंदी वाक्य नहीं मिले; मूल अंश अंग्रेज़ी में रखा गया।',
        'empty_response': 'स्थानीय मॉडल ने खाली उत्तर दिया।',
        'local_model_error': 'स्थानीय अनुवाद मॉडल का उत्तर उपयोग नहीं हो सका।',
        'partial_translation': 'बाकी पाठ-अंश अंग्रेज़ी में रखे गए हैं।',
        'FileNotFoundError': 'स्थानीय अनुवाद मॉडल उपलब्ध नहीं है।',
        'ImportError': 'स्थानीय अनुवाद मॉडल की निर्भरता उपलब्ध नहीं है।',
    }
    if status in captions:
        st.caption(' '.join(filter(None, [captions[status], reasons.get(info.get('reason'))])))
    with st.expander('हिंदी अनुवाद की स्थिति'):
        safe_fields = (
            'provider', 'model', 'retry_model', 'version', 'status', 'reason',
            'request_chunks', 'failure_scope', 'source_fragment_kind',
            'structure_error', 'structure_error_counts', 'translated_chunks',
            'failed_chunks', 'failure_reason_counts', 'remaining_latin_count',
            'remaining_latin_words',
        )
        st.json({key: info[key] for key in safe_fields if key in info})
