"""Final-answer English-to-Hindi translation with the local IndicTrans2 model."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import os
import re
import sys
MODEL = 'ai4bharat/indictrans2-en-indic-dist-200M'
TRANSLATION_VERSION = 'indictrans2-200m-v1'


class TranslationValidationError(ValueError):
    pass

# Preserve structured links, code, quantities, units and formulae as indivisible tokens.
FORMULA = r'\b(?:ZnSO\s*4|MnSO\s*4|FeSO\s*4|P2O5|K2O|NPK|NAA|FYM|SSP|DAP|DAS|Zn|S|N|P|K)\b'
UNIT = r'(?:kg|g|mg|ml|mL|L|litres?|liters?|tonnes?|tons?|t|cm|mm|m|ha|acres?|hectares?|ppm|%|°C)(?:\s*/\s*(?:ha|hectare|acre|L|litre|liter))?'
NUMBER = r'\d+(?:[.,:/–-]\d+)*'
PROTECTED = re.compile(
    r'```[\s\S]*?```|`[^`\n]+`|\[[^\]\n]*\]\([^)]+\)|https?://[^\s<>\)]+|\*\*|__|\||'
    + NUMBER + r'(?:\s*' + UNIT + r'\b)?(?:\s*' + FORMULA + r')?|'
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
    """Translate English prose locally and retain independently translated fragments.

    Translation uses the locally loaded IndicTrans2 200M checkpoint. Answer
    fragments are not sent to a translation API. Numeric/formula checks do not
    establish semantic correctness; translation still needs evaluation.
    """
    existing = result.get('translation')
    if (isinstance(existing, dict) and existing.get('provider') == 'local'
            and existing.get('status') in {'translated', 'partial', 'not_needed'}):
        return result
    original = str(result.get('answer', ''))
    info = {'provider': 'local', 'model': MODEL, 'target_language': 'hi-IN',
            'version': TRANSLATION_VERSION}
    def finish(answer, status, reason=None, **details):
        metadata = {**info, 'status': status, **details}
        if reason:
            metadata['reason'] = reason
        return {**result, 'answer': answer, 'translation': metadata}
    if _setting('KISAANAI_HINDI_TRANSLATION', '1').lower() in {'0', 'false', 'no'}:
        return finish(original, 'disabled')
    unavailable = 'इस उत्तर का पूरा हिंदी अनुवाद अभी उपलब्ध नहीं है। कृपया फिर प्रयास करें।'
    if TOKEN.search(original):
        return finish(unavailable, 'fallback', 'reserved_token_in_source')
    protected, values = _protect(original, _protected_names(result))
    segments = list(NON_HINDI.finditer(protected))
    # Formula-only passages need no model; keep approved scientific notation.
    pending = [m for m in segments if re.search(r'[A-Za-z]', TOKEN.sub('', m.group()))]
    if not pending:
        return finish(original, 'not_needed')
    hf_token = _setting('HF_TOKEN')
    if not hf_token:
        return finish(unavailable, 'not_configured', 'HF_TOKEN')
    replacements = []
    failures = []
    for match in pending:
        segment = match.group()
        # Keep Markdown list/header markers and surrounding whitespace local.
        prefix = re.match(r'^[\s#>*+\-]*', segment).group()
        suffix = re.search(r'\s*$', segment).group()
        core = segment[len(prefix):len(segment) - len(suffix) if suffix else None]
        translated_chunks = []
        for chunk in _chunks(core):
            try:
                translated_chunks.append(_translate(chunk, hf_token))
            except Exception as exc:
                translated_chunks.append(chunk)
                reason = (str(exc) if isinstance(exc, TranslationValidationError)
                          else type(exc).__name__)
                failures.append(reason)
        replacements.append((
            match.start(), match.end(),
            prefix + ' '.join(translated_chunks) + suffix,
        ))
    output = protected
    for start, end, translated_text in reversed(replacements):
        output = output[:start] + translated_text + output[end:]
    if Counter(TOKEN.findall(output)) != Counter(values.keys()):
        return finish(unavailable, 'fallback', 'protected_values_lost')
    for token, value in values.items():
        output = output.replace(token, value)
    if failures:
        # Do not show a mixed-language answer. Keep translation all-or-nothing at
        # the user-visible answer level while retaining failure details in metadata.
        return finish(unavailable, 'fallback', 'fragment_translation_failed',
                      failed_fragments=len(failures), failure_reason=failures[0])
    return finish(output, 'translated', translated_fragments=len(pending))
