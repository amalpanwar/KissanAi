"""Final-answer English-to-Hindi translation with the local IndicTrans2 model."""
from __future__ import annotations

from functools import lru_cache
import os
import re
import sys
MODEL = 'ai4bharat/indictrans2-en-indic-dist-200M'
TRANSLATION_VERSION = 'indictrans2-200m-v1'


class TranslationValidationError(ValueError):
    pass

# Preserve structured links, code, quantities, units and formulae as indivisible tokens.
FORMULA = r'\b(?:ZnSO\s*4|MnSO\s*4|FeSO\s*4|P\s*[2₂]\s*O\s*[5₅]|K\s*[2₂]\s*O|NPK|NAA|FYM|SSP|DAP|DAS|Zn|S|N|P|K)\b'
UNIT = r'(?:kg|g|mg|ml|mL|L|litres?|liters?|tonnes?|tons?|t|cm|mm|m|ha|acres?|hectares?|ppm|%|°C|quintals?)(?:\s*/\s*(?:kg|g|mg|ml|mL|L|litres?|liters?|ha|hectare|acre))?'
NUMBER = r'\d+(?:[.,:/–-]\d+)*'
PROTECTED = re.compile(
    r'\[PDF page \d+\]|```[\s\S]*?```|`[^`\n]+`|\[[^\]\n]*\]\([^)]+\)|https?://[^\s<>\)]+|\*\*|__|\||'
    + NUMBER + r'(?:\s*' + UNIT + r'(?![A-Za-z]))?(?:\s*' + FORMULA + r')?(?:\s*' + UNIT + r'(?![A-Za-z]))?|'
    + FORMULA + r'|(?i:\b(?:Carbendazim|Thiram|Thir\s+am|Carboxin|Phorate|Carbofuran|Endosulfan|Methyl\s+de\s+meton|Dimethoat|Isoproturon|Azospirillum|Phosphobacteria|Azophos|TNAU|PE|COW\s*\(W\)\s*\d+|CO\s*\d+|Samba\s+Wheat|Pennisetum\s+glaucum)\b)'
)
NON_HINDI = re.compile(r'[^\u0900-\u097f\n]+')
TOKEN = re.compile(r'ZXQ[A-Z]+QXZ')


def _setting(name: str, default: str = '') -> str:
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    # Streamlit secrets are read only when Streamlit is already in use.
    st = sys.modules.get('streamlit')
    if st is not None:
        try:
            return str(st.secrets.get(name, default)).strip()
        except Exception:
            pass
    return default


def _token(index: int) -> str:
    label = ''
    while True:
        label = chr(65 + index % 26) + label
        index = index // 26 - 1
        if index < 0:
            return 'ZXQ' + label + 'QXZ'


def _protected_names(result: dict) -> list[str]:
    """Use explicit entity metadata, never guess names from capitalized prose."""
    names = list(result.get('protected_names') or [])
    keys = ('place', 'sub_district', 'district', 'state', 'city', 'name', 'person_name')
    def add_location(location):
        if isinstance(location, dict):
            names.extend(location.get(k) for k in keys)
    add_location(result.get('location'))
    for message in (result.get('agent_trace') or {}).get('messages', []):
        payload = message.get('payload', {})
        add_location((payload.get('evidence') or {}).get('location'))
        # Context is supplied by the app's location controls. It is not sent.
        context = payload.get('context', '')
        names.extend(re.findall(
            r'(?:^|\|)\s*(?:State|District|Subdistrict|Place|City|Name|नाम|जिला|स्थान)\s*:\s*([^|]+)',
            context, re.I,
        ))
    # Preserve the complete resolved weather location, including geocoder names
    # absent from the local lookup table and legacy weather responses.
    names.extend(re.findall(
        r'(?:मौसम|पूर्वानुमान|बारिश)[^\n()]*\(([^()\n]+)\)',
        str(result.get('answer', '')),
    ))
    return sorted({n.strip() for n in names if isinstance(n, str) and n.strip()},
                  key=len, reverse=True)


def _protect(text: str, names: list[str] | None = None) -> tuple[str, dict[str, str]]:
    values = {}
    def replace(match):
        token = _token(len(values))
        values[token] = match.group()
        return token
    # Names are matched before numbers/formulae; longest names win. Restore the
    # exact matched spelling after translation, including case and spaces.
    entity_pattern = '|'.join(r'(?<!\w)' + re.escape(n) + r'(?!\w)' for n in (names or []))
    pattern = re.compile(entity_pattern + '|' + PROTECTED.pattern) if entity_pattern else PROTECTED
    return pattern.sub(replace, text), values


def _chunks(text: str, limit: int = 1800) -> list[str]:
    chunks = []
    while len(text) > limit:
        boundary = text.rfind(' ', 0, limit + 1)
        if boundary <= 0:
            raise ValueError('Oversized translation token')
        chunks.append(text[:boundary])
        text = text[boundary + 1:]
    if text:
        chunks.append(text)
    return chunks

def _sentence_chunks(text: str, limit: int = 700) -> list[str]:
    """Keep IndicTrans inputs short enough to isolate a bad sentence or placeholder."""
    sentences = re.split(r'(?<=[.!?])\\s+(?=[A-Z0-9])', text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_chunks(sentence, limit))
        elif not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def _protected_pattern(names: list[str] | None = None) -> re.Pattern:
    """Keep technical entities and structure out of the MT model input."""
    name_patterns = [r'(?<!\w)' + re.escape(name) + r'(?!\w)'
                     for name in (names or []) if name]
    structural = r'(?m:^[ \t]*(?:#{1,6}[ \t]*|[-+*][ \t]+|\d+[.)][ \t]+))|\n+|\[(?:PDF\s+)?page\s+\d+\]|\[\d+\]'
    return re.compile('|'.join(name_patterns + [PROTECTED.pattern, structural]), re.IGNORECASE)
@lru_cache(maxsize=1)
def _load_model(hf_token: str):
    """Load the translation model once per process; inference stays on-host."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, token=hf_token or None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL, trust_remote_code=True, token=hf_token or None,
    ).to(device)
    model.eval()
    return torch, tokenizer, model, IndicProcessor(inference=True), device


@lru_cache(maxsize=256)
def _translate(text: str, hf_token: str = '') -> str:
    torch, tokenizer, model, processor, device = _load_model(hf_token)
    prepared = processor.preprocess_batch(
        [text], src_lang='eng_Latn', tgt_lang='hin_Deva',
    )
    inputs = tokenizer(
        prepared, truncation=True, padding='longest',
        return_tensors='pt', return_attention_mask=True,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, use_cache=True, min_length=0, max_length=512,
            num_beams=4, num_return_sequences=1,
        )
    decoded = tokenizer.batch_decode(
        generated, skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    outputs = processor.postprocess_batch(decoded, lang='hin_Deva')
    output = outputs[0] if outputs else ''
    if not isinstance(output, str) or not output.strip():
        raise TranslationValidationError('empty_response')
    # Each protected item must occur exactly once and stay in source order.
    if TOKEN.findall(output) != TOKEN.findall(text):
        raise TranslationValidationError('protected_values_changed')
    residual = TOKEN.sub('', output)
    if re.search(r'\d', residual) or re.search(r'[A-Za-z]', residual):
        raise TranslationValidationError('untranslated_text_or_new_numbers')
    if not re.search(r'[\u0900-\u097f]', residual):
        raise TranslationValidationError('hindi_missing')
    return output.strip()


def translate_answer(result: dict) -> dict:
    """Translate prose while keeping values, names, citations, and formatting external.

    Failures are isolated per prose fragment. Successfully translated text remains
    Hindi; an untranslated fragment stays visible and is disclosed in metadata.
    """
    existing = result.get('translation')
    if (isinstance(existing, dict) and existing.get('provider') == 'local'
            and existing.get('status') in {'translated', 'partial', 'not_needed', 'source_answer_fallback'}):
        return result
    original = str(result.get('answer', ''))
    info = {'provider': 'local', 'model': MODEL, 'target_language': 'hi-IN',
            'version': TRANSLATION_VERSION}
    def finish(answer, status, reason=None, **details):
        metadata = {**info, 'status': status, **details}
        if reason:
            metadata['reason'] = reason
        return {**result, 'answer': answer, 'translation': metadata}
    unavailable = 'इस उत्तर का पूरा हिंदी अनुवाद अभी उपलब्ध नहीं है। कृपया फिर प्रयास करें।'
    def failed_translation(reason, status='fallback', **details):
        if original.strip():
            if result.get('topic') in {'crop_guide', 'crop_guide_followup'}:
                return finish(original, 'source_answer_fallback', reason,
                              answer_language='en-IN', **details)
            return finish(original, status, reason, **details)
        return finish(unavailable, status, reason, **details)
    if _setting('KISAANAI_HINDI_TRANSLATION', '1').lower() in {'0', 'false', 'no'}:
        return failed_translation('disabled', 'disabled')
    if TOKEN.search(original):
        return failed_translation('reserved_token_in_source')

    pattern = _protected_pattern(_protected_names(result))
    spans = list(pattern.finditer(original))
    prose = []
    cursor = 0
    for match in spans:
        if match.start() > cursor:
            prose.append((cursor, match.start(), original[cursor:match.start()]))
        cursor = match.end()
    if cursor < len(original):
        prose.append((cursor, len(original), original[cursor:]))
    pending = [(start, end, fragment) for start, end, fragment in prose
               if re.search(r'[A-Za-z]', fragment)]
    if not pending:
        return finish(original, 'not_needed')

    hf_token = _setting('HF_TOKEN')
    if not hf_token:
        return failed_translation('HF_TOKEN', 'not_configured')

    output_parts = []
    failed_fragments = []
    translated_count = 0
    cursor = 0
    for start, end, fragment in pending:
        output_parts.append(original[cursor:start])
        leading = fragment[:len(fragment) - len(fragment.lstrip())]
        trailing = fragment[len(fragment.rstrip()):]
        core = fragment[len(leading):len(fragment) - len(trailing) if trailing else None]
        translated_chunks = []
        for chunk in _sentence_chunks(core):
            if not re.search(r'[A-Za-z]', chunk):
                translated_chunks.append(chunk)
                continue
            try:
                translated_chunks.append(_translate(chunk, hf_token))
                translated_count += 1
            except Exception as exc:
                translated_chunks.append(chunk)
                reason = (str(exc) if isinstance(exc, TranslationValidationError)
                          else type(exc).__name__)
                failed_fragments.append({'text': chunk, 'reason': reason})
        output_parts.append(leading + ' '.join(translated_chunks) + trailing)
        cursor = end
    output_parts.append(original[cursor:])
    output = ''.join(output_parts)
    if failed_fragments:
        status = 'partial' if translated_count else (
            'source_answer_fallback' if result.get('topic') in {'crop_guide', 'crop_guide_followup'} else 'fallback'
        )
        metadata = {
            **info, 'status': status,
            'reason': failed_fragments[0]['reason'],
            'failure_reason': failed_fragments[0]['reason'],
            'failed_fragments': len(failed_fragments),
            'untranslated_fragments': [
                {'text': item['text'][:500], 'reason': item['reason']}
                for item in failed_fragments[:10]
            ],
        }
        if status == 'source_answer_fallback':
            metadata['answer_language'] = 'en-IN'
        elif status == 'partial':
            metadata['answer_language'] = 'mixed'
        return {**result, 'answer': output, 'translation': metadata}
    return finish(output, 'translated', translated_fragments=translated_count)