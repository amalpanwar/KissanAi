"""Discover repository PDFs omitted from the saved index; cache text discovery.

A lexical shortlist bounds encoding work on a cold deployment. The shared semantic cross-encoder decides relevance; no crop-specific routing.
"""
from functools import lru_cache
from pathlib import Path
from app.chunking import split_text
from app.pdf_extract import read_pdf_pages

ROOT = Path(__file__).resolve().parents[1] / 'data/raw'


def missing_pdfs(indexed_sources):
    indexed = {Path(str(path)).resolve() for path in indexed_sources}
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                 for path in sorted(ROOT.rglob('*.pdf')) if path.resolve() not in indexed)


@lru_cache(maxsize=2)
def catalog(signature):
    from app.multilingual_retrieval import language
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows=[]; failed=[]
    for filename, _, _ in signature:
        try:
            pages=read_pdf_pages(Path(filename), prefer_docling=False)
            for number,page in enumerate(pages,1):
                for text in split_text(page, chunk_size=1400, overlap=180):
                    rows.append({'text':text, 'source_file':str(Path(filename).relative_to(ROOT.parents[1])), 'page':number, 'language':language(text)})
        except Exception:
            failed.append(filename)
    partitions={}
    for lang in {row['language'] for row in rows}:
        subset=[row for row in rows if row['language']==lang]
        vectorizer=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', sublinear_tf=True)
        try:
            matrix=vectorizer.fit_transform(row['text'] for row in subset)
            partitions[lang]=(subset,vectorizer,matrix)
        except ValueError:
            continue
    return partitions,failed


def candidates(advisor,payload,indexed_sources):
    signature=missing_pdfs(indexed_sources)
    if not signature:
        return [], {'files':0,'failures':[]}
    partitions,failed=catalog(signature)
    rows=[]
    for lang,(subset,vectorizer,matrix) in partitions.items():
        query=(payload.get('query_variants') or {}).get(lang)
        if not query:
            continue
        explicit=advisor._extract_crop_from_query(payload['question'])
        crop=explicit or advisor._extract_preferred_crop_from_context(payload.get('context',''))
        query=query+(f'\nCrop context: {crop}' if crop else '')
        lexical=(matrix @ vectorizer.transform([query]).T).toarray().ravel()
        # Source diversity prevents a large pesticide PDF hiding a crop guide.
        selected=[]
        for source in {row['source_file'] for row in subset}:
            ids=[i for i,row in enumerate(subset) if row['source_file']==source and lexical[i]>0]
            selected.extend(sorted(ids,key=lambda i:-lexical[i])[:40])
        if not selected:
            continue
        for i in selected:
            rows.append(dict(subset[i], score=float(lexical[i]), search_query=query))
    return rows, {'files':len(signature),'failures':failed}
