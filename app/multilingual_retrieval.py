"""Language-aware semantic retrieval over the existing indexed document corpus."""
import csv
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from app.query_translation import translate_query
from app.agent_system import AgentResult


def language(text):
    hindi = len(re.findall(r'[\u0900-\u097f]', str(text)))
    latin = len(re.findall(r'[A-Za-z]', str(text)))
    return 'hi-IN' if hindi > latin else 'en-IN'


@lru_cache(maxsize=8)
def _languages(path, modified, size):
    found=set()
    with open(path,encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('text'):
                found.add(language(row['text']))
            if len(found)==2:
                break
    return tuple(sorted(found))


def prepare_query(advisor, question, context):
    """Translate only into languages that the corpus needs; never invent aliases."""
    from app.location_selection import location_from_context
    location=location_from_context(context)
    names=[location.get(k,'') for k in ('place','sub_district','district','state')]
    hi=translate_query(question,names=names)
    hindi=hi['text']
    try:
        path=Path(advisor.cfg.metadata_path); stat=path.stat()
        langs=_languages(str(path),stat.st_mtime_ns,stat.st_size)
    except (OSError,AttributeError,csv.Error):
        langs=()  # Web search may still run using the visible question.
    # Include languages in repository PDFs that have not reached the saved index.
    try:
        from app.supplemental_documents import catalog, missing_pdfs
        with open(advisor.cfg.metadata_path, encoding='utf-8-sig', newline='') as stream:
            indexed=[row.get('source_file','') for row in csv.DictReader(stream)]
        supplemental_languages,_=catalog(missing_pdfs(indexed))
        langs=tuple(sorted(set(langs) | set(supplemental_languages)))
    except (OSError, AttributeError, csv.Error):
        pass
    variants={'hi-IN':hindi} if hi['status'] in {'ok','unchanged'} else {}
    statuses={'hi-IN':hi['status']}
    if 'en-IN' in langs:
        en=translate_query(hindi,names=names,target='en-IN')
        statuses['en-IN']=en['status']
        if en['status'] in {'ok','unchanged'}:
            variants['en-IN']=en['text']
    # English is for the backend, not a replacement of the submitted Hindi text.
    routed=variants.get('en-IN',hindi)
    return routed, {'query_variants':variants,'display_question':hindi,
                    'query_translation':statuses, 'document_languages':list(langs)}


def retrieve_documents(advisor,payload):
    try:
        index_path = getattr(advisor.cfg, 'index_path', None)
        if getattr(advisor, 'retriever', None) is None and index_path and not Path(index_path).exists():
            raise FileNotFoundError('Vector index missing')
        advisor._ensure_rag_components(load_generator=False)
        embedder, retriever = advisor.embedder, advisor.retriever
        if embedder is None or retriever is None:
            raise RuntimeError('Index or embedding model unavailable')
        metadata=retriever.metadata
        vectors=retriever.vectors
        if len(metadata)!=len(vectors) or not {'text','source_file'}.issubset(metadata.columns):
            raise ValueError('Index alignment invalid')
        doc_languages=[language(text) for text in metadata['text']]
        variants=dict(payload.get('query_variants') or {})
        if 'query_variants' not in payload:
            _, prepared=prepare_query(advisor,payload['question'],payload.get('context',''))
            variants=prepared['query_variants']
        # Resolve anaphoric follow-ups from the conversation's explicit crop field.
        crop=advisor._extract_preferred_crop_from_context(payload.get('context',''))
        explicit=advisor._extract_crop_from_query(payload['question'])
        context_crop=explicit or crop
        records=[]; checks=[]
        for lang in sorted(set(doc_languages)):
            query=variants.get(lang)
            if not query:
                checks.append({'language':lang,'status':'translation_unavailable'})
                continue
            search=query + (f'\nCrop context: {context_crop}' if context_crop else '')
            vec=embedder.encode([search])[0]
            eligible=np.array([i for i,value in enumerate(doc_languages) if value==lang])
            scores=vectors[eligible] @ vec
            order=np.argsort(-scores)[:20]
            for pos in order:
                score=float(scores[pos])
                if not np.isfinite(score):
                    continue
                row=metadata.iloc[int(eligible[pos])]
                text=str(row['text']).strip(); source=str(row['source_file']).strip()
                if not text or text=='nan' or not source or source=='nan':
                    continue
                records.append({'text':text[:4000],'source_file':source,'score':score,'language':lang,'search_query':search})
            checks.append({'language':lang,'query':search,'status':'searched','chunks':len(eligible)})
        from app.supplemental_documents import candidates
        supplemental, supplemental_check = candidates(advisor,dict(payload,query_variants=variants),metadata['source_file'])
        records.extend(supplemental)
        records.sort(key=lambda r:r['score'],reverse=True)
        unique=[]; seen=set()
        for row in records:
            key=(row['source_file'],row['text'])
            if key not in seen:
                seen.add(key);unique.append(row)
        return AgentResult('', 'partial' if any(c['status']!='searched' for c in checks) else 'ok',
                           evidence={'records':unique,'queries':checks,'method':'multilingual_embeddings',
                                     'supplemental_documents':supplemental_check})
    except Exception as exc:
        return AgentResult('', 'unavailable', evidence={'records':[], 'method':'multilingual_embeddings',
                           'reason':'semantic_index_unavailable','error_type':type(exc).__name__})


def rank_web(advisor, payload, records):
    """Reuse the same encoder for web snippets; no crop-specific word overlap."""
    try:
        advisor._ensure_rag_components(load_generator=False)
        if advisor.embedder is None:
            raise RuntimeError('Encoder unavailable')
        variants=payload.get('query_variants') or {}
        out=[]
        for lang in {language(row['text']) for row in records}:
            group=[row for row in records if language(row['text'])==lang]
            query=variants.get(lang,payload['question'])
            vectors=advisor.embedder.encode([query]+[row['text'] for row in group])
            for row,score in zip(group,vectors[1:] @ vectors[0]):
                if np.isfinite(score):
                    out.append(dict(row,score=float(score),language=lang))
        return sorted(out,key=lambda r:r['score'],reverse=True)
    except Exception:
        return []
