from __future__ import annotations


SYSTEM_PROMPT = """
आप एक कृषि सहायक हैं जो पश्चिमी उत्तर प्रदेश (Western Uttar Pradesh) के किसानों के लिए सलाह देता है।
हमेशा क्षेत्र, मौसम, बजट, सिंचाई, मिट्टी और जोखिम का ध्यान रखकर उत्तर दें।
अगर जानकारी संदर्भ में नहीं है तो स्पष्ट बताएं और अनुमान न लगाएं।
उत्तर हिंदी में दें, और जरूरी तकनीकी शब्द सरल भाषा में समझाएं।
""".strip()


PROMPT_STRUCTURES = {
    "profitability_reasoning": [
        "0) एक पंक्ति में: समझा गया सवाल (हिंदी)",
        "1) सबसे उपयुक्त विकल्प",
        "2) अपेक्षित लागत व संभावित लाभ",
        "3) सर्वोत्तम उत्पादन/निर्णय के लिए जरूरी शर्तें",
        "4) जोखिम और बचाव",
    ],
    "pesticide_lookup": [
        "0) एक पंक्ति में: समझा गया सवाल (हिंदी)",
        "1) सबसे संभावित रोग/कीट या समस्या",
        "2) दवा/उपाय विकल्प, dose और formulation",
        "3) पानी/घोल, PHI या seed-treatment जैसी जरूरी सावधानियां",
        "4) अगर डेटा कमजोर हो तो स्पष्ट सीमा बताएं",
    ],
    "official_notice": [
        "0) एक पंक्ति में: समझा गया सवाल (हिंदी)",
        "1) policy/scheme/notification का सीधा मतलब",
        "2) अगर rate/value है तो वही साफ लिखें",
        "3) यह किस पर लागू होता है",
        "4) जरूरी caveat या date/rule",
    ],
    "general_explainer": [
        "0) एक पंक्ति में: समझा गया सवाल (हिंदी)",
        "1) सीधा उत्तर",
        "2) स्रोत-संबंधित मुख्य बिंदु",
        "3) practical implication",
        "4) risk/caveat अगर जरूरी हो",
    ],
    "general_factoid": [
        "0) एक पंक्ति में: समझा गया सवाल (हिंदी)",
        "1) सीधा और संक्षिप्त उत्तर",
        "2) जरूरी supporting detail",
        "3) source-based caution अगर जरूरी हो",
    ],
}


def build_prompt(
    user_query: str,
    context_chunks: list[dict],
    *,
    query_family: str = "general_explainer",
    max_chunks: int = 3,
) -> str:
    trimmed_context = context_chunks[: max(1, int(max_chunks))]
    context_text = "\n\n".join(
        [
            f"[Source: {c.get('source_file', 'unknown')}]\n{(c.get('text', '') or '')[:350]}"
            for c in trimmed_context
        ]
    )
    structure_lines = PROMPT_STRUCTURES.get(query_family) or PROMPT_STRUCTURES["general_explainer"]
    structure_text = "\n".join(structure_lines)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"संदर्भ जानकारी:\n{context_text}\n\n"
        f"किसान का सवाल (हिंदी में समझा गया): {user_query}\n\n"
        "उत्तर संरचना:\n"
        f"{structure_text}\n"
    )
