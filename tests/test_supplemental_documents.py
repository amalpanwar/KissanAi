import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from app import supplemental_documents as docs


class SupplementalDocumentTests(unittest.TestCase):
    def test_discovers_only_missing_pdfs_and_preserves_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data/raw';root.mkdir(parents=True)
            known=root/'known.pdf';known.touch()
            missing=root/'guide.pdf';missing.touch()
            advisor=SimpleNamespace(_extract_crop_from_query=lambda q:None,_extract_preferred_crop_from_context=lambda c:None)
            with patch.object(docs,'ROOT',root),patch.object(docs,'read_pdf_pages',return_value=['Rice varieties include early and late cultivars.']) as read:
                signature=docs.missing_pdfs([str(known)])
                self.assertEqual([x[0] for x in signature],[str(missing)])
                rows,checks=docs.candidates(advisor,{'question':'rice varieties','query_variants':{'en-IN':'rice varieties'}},[str(known)])
                self.assertEqual(rows[0]['page'],1)
                self.assertEqual(rows[0]['source_file'],'data/raw/guide.pdf')
                self.assertEqual(checks['failures'],[])
                docs.candidates(advisor,{'question':'rice','query_variants':{'en-IN':'rice'}},[str(known)])
                read.assert_called_once()
            docs.catalog.cache_clear()
