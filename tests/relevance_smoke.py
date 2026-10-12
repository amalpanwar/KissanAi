"""Manual real-model regression; requires cached/downloadable model weights."""
from pathlib import Path
import numpy as np
import pandas as pd
from app.embeddings import Embedder
from app.evidence_reranker import model, bibliography
import torch
metadata=pd.read_csv('data/processed/chunk_metadata.csv')
vectors=np.load('data/processed/vector_index.npz')['vectors']
embedder=Embedder('intfloat/multilingual-e5-small')
for question in ['Which type of rice is better?', 'What rice varieties should I grow?', 'How do I get my soil tested?']:
    similarities=vectors @ embedder.encode([question])[0]
    ids=np.argsort(-similarities)[:20]
    rows=[metadata.iloc[i].to_dict() for i in ids if not bibliography(str(metadata.iloc[i]['text']))]
    scores=model().predict([(question,str(row['text'])) for row in rows],activation_fct=torch.nn.Identity())
    print(question, flush=True)
    for score,row in sorted(zip(scores,rows),key=lambda r:-r[0])[:5]:
        print(round(float(score),3),row['source_file'],str(row['text'])[:350],flush=True)

from types import SimpleNamespace
from app.supplemental_documents import candidates
from app.evidence_reranker import verify
advisor=SimpleNamespace(embedder=embedder,_extract_crop_from_query=lambda q:'Rice',_extract_preferred_crop_from_context=lambda c:None)
question='What rice varieties should I grow?'
payload={'question':question,'query_variants':{'en-IN':question}}
rows,checks=candidates(advisor,payload,metadata.source_file)
accepted,status=verify(payload,rows)
assert status=='ok'
assert any('Crop Production guide.pdf' in r['source_file'] and r['page']==13 for r in accepted)
assert all('herbicide' not in r['source_file'] for r in accepted)
reported={'question':'Which type of rice is better?','query_variants':{'en-IN':'Which type of rice is better?'}}
herbicides=metadata[metadata.source_file.str.contains('herbicides')].head(10).to_dict(orient='records')
assert verify(reported,herbicides)==([], 'ok')
print('Real-model regression passed: rice variety table found; unrelated herbicide excerpts rejected.')
