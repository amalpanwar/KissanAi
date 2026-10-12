"""Model-free rendering of the existing sugarcane source guide and trench extraction."""
import json
import logging
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRENCH = ROOT / 'data/processed/trench_planting_sugarcane_structured.json'


def sugarcane_guide(advisor, question, context=''):
    normalized = advisor._normalize_hinglish(question)
    crop = (advisor._extract_crop_from_query(normalized)
            or advisor._extract_preferred_crop_from_context(context))
    if crop != 'Sugarcane' or advisor._has_profitability_terms(normalized):
        return None
    from app.agent_system import Coordinator
    if any(name != 'agronomy' for name in Coordinator.domains(normalized)):
        return None
    trench = bool(re.search(r'\b(?:trench|ssnm)\b|ट्रेंच|खाई', normalized, re.I))
    general = advisor._has_crop_guide_terms(normalized) or bool(re.search(r'\bfarming\b', normalized, re.I))
    if not trench and re.search(r'fertilizer|irrigation|variet|खाद|उर्वरक|सिंचाई|किस्म', normalized, re.I):
        return None
    if not (general or trench):
        return None
    parts, references = [], []
    guide_found = trench_found = False
    try:
        from app.crop_guide import build_crop_production_guide
        if general:
            answer, sources = build_crop_production_guide('Sugarcane')
            if answer:
                guide_found = True
                parts.append(answer)
                references.extend(sources)
                parts.append('यह सामान्य खेती की जानकारी फसल उत्पादन गाइड से है; क्षेत्र और मिट्टी के अनुसार स्थानीय उपयुक्तता की जाँच करें।')
        # This extraction is already in Hindi and verified against the supplied PDF.
        data = json.loads(TRENCH.read_text(encoding='utf-8')) if TRENCH.exists() else {}
        source = data.get('source_file', '')
        if (data.get('crop') == 'Sugarcane' and data.get('source_type') == 'reviewed_structured_extraction'
                and source and (ROOT / source).is_file()):
            points = [p for p in data.get('implementation_hi', []) if isinstance(p, str) and p.strip()]
            if points:
                trench_found = True
                parts.append('ट्रेंच विधि — आपके पूरक अध्ययन के अनुसार:\n' + '\n'.join('- '+p for p in points))
                parts.append(str(data.get('source_context_hi', '')))
                references.append(source)
                if re.search(r'formula|ssnm|फॉर्मूला|सूत्र', normalized, re.I):
                    from app.pdf_extract import read_pdf_pages
                    text = '\n'.join(read_pdf_pages(ROOT / source, prefer_docling=False)[:3])
                    row = re.search(r'Kg/ha\s+' + r'\s+'.join([r'(\d+(?:\.\d+)?)'] * 7), text)
                    target = re.search(r'target\s+yield\s+(\d+)\s*q/ha', text, re.I)
                    if row and target:
                        labels = ['नाइट्रोजन', 'फॉस्फोरस', 'पोटैशियम', 'बोरॉन', 'गंधक', 'लोहा', 'जिंक']
                        parts.append(f"अध्ययन में {target.group(1)} क्विंटल/हेक्टेयर लक्ष्य उपज के लिए पोषक अवशोषण (किग्रा/हेक्टेयर):\n"
                                     + '\n'.join(f'- {label}: {value}' for label, value in zip(labels, row.groups()))
                                     + '\nये अध्ययन के पोषक अवशोषण मान हैं। इन्हें सीधे उर्वरक की मात्रा न मानें; उर्वरक की मात्रा मिट्टी की जाँच और पोषक प्रतिशत पर निर्भर है।')
                    else:
                        parts.append('इस दस्तावेज़ से पोषक तालिका साफ नहीं पढ़ी जा सकी; कोई मात्रा अनुमान से नहीं दी गई है।')
        if not parts:
            return {'answer': 'गन्ने की खेती की स्रोत गाइड उपलब्ध नहीं है। कृपया फसल उत्पादन और ट्रेंच दस्तावेज़ की उपलब्धता जाँचें।',
                    'references': [], 'retrieved': [], 'topic': 'crop_guide', 'agent_status': 'unavailable'}
        if general and not guide_found:
            parts.insert(0, 'सामान्य खेती की गाइड उपलब्ध नहीं है; केवल उपलब्ध ट्रेंच स्रोत की जानकारी दी गई है।')
        if trench and not trench_found:
            parts.append('ट्रेंच दस्तावेज़ की सत्यापित जानकारी उपलब्ध नहीं है।')
    except Exception:
        logging.getLogger(__name__).exception('Sugarcane source guide failed')
        return {'answer': 'गन्ने की स्रोत गाइड पढ़ते समय समस्या आई। कृपया दस्तावेज़ की उपलब्धता जाँचें।',
                'references': [], 'retrieved': [], 'topic': 'crop_guide', 'agent_status': 'unavailable'}
    status = 'partial' if (general and not guide_found) or (trench and not trench_found) else 'ok'
    return {'answer': '\n\n'.join(parts), 'references': list(dict.fromkeys(references)),
            'retrieved': [], 'topic': 'crop_guide', 'agent_status': status,
            'translation': {'provider': 'source_renderer', 'status': 'not_needed', 'version': 'sugarcane-source-v1'},
            'agent_trace': {'plan': ['agronomy'], 'planner': 'source_guide', 'calls': 0,
                            'display_question': question, 'backend_question': 'Sugarcane',
                            'decisions': [{'agent': 'agronomy', 'status': 'ok',
                                           'decision': 'Render existing crop guide and verified trench extraction'}]}}
