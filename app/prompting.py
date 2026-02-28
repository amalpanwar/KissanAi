from __future__ import annotations


SYSTEM_PROMPT = """
आप एक कृषि सहायक हैं जो पश्चिमी उत्तर प्रदेश (Western Uttar Pradesh) के किसानों के लिए सलाह देता है।
हमेशा क्षेत्र, मौसम, बजट, सिंचाई, मिट्टी और जोखिम का ध्यान रखकर उत्तर दें।
अगर जानकारी संदर्भ में नहीं है तो स्पष्ट बताएं और अनुमान न लगाएं।
उत्तर हिंदी में दें, और जरूरी तकनीकी शब्द सरल भाषा में समझाएं।
""".strip()


def build_prompt(user_query: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        [
            f"[Source: {c.get('source_file', 'unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        ]
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"संदर्भ जानकारी:\n{context_text}\n\n"
        f"किसान का सवाल: {user_query}\n\n"
        "उत्तर संरचना:\n"
        "1) सबसे उपयुक्त फसल विकल्प\n"
        "2) अपेक्षित लागत व संभावित लाभ\n"
        "3) सर्वोत्तम उत्पादन के लिए जरूरी शर्तें\n"
        "4) जोखिम और बचाव\n"
    )
