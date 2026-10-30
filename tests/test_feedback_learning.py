import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.db import (init_db, get_conn, create_query_log, save_feedback, review_feedback,
                    export_training_feedback, get_training_feedback_examples)
from app.feedback import validate_feedback_with_local_sources
from app.feedback_learning import build_feedback_dataset
from app.symptom_matcher import load_feedback_rows


class FeedbackLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name) / 'feedback.db')
        init_db(self.db)
        with get_conn(self.db) as conn:
            conn.execute("INSERT INTO users(id,username,password_hash,role) VALUES(1,'admin','test','admin')")
            conn.execute("INSERT INTO users(id,username,password_hash,role) VALUES(2,'farmer','test','user')")

    def add(self, question='गेहूं की खेती कैसे करें?', topic='crop_guide', correction='बुवाई से पहले मिट्टी की जांच कराएं।'):
        qid = create_query_log(self.db, dict(user_id=2, user_query=question, answer_text='Old answer', topic=topic))
        return save_feedback(self.db, dict(query_log_id=qid, user_id=2, rating='provide_correction',
                             correction_text=correction, validation_status='source_matched', is_training_eligible=1))

    def test_only_reviewed_corrections_are_exported_and_rejection_removes_memory(self):
        fid = self.add()
        self.assertEqual(get_training_feedback_examples(self.db), [])
        with self.assertRaises(ValueError):
            review_feedback(self.db, fid, 2, 'accepted', True)
        review_feedback(self.db, fid, 1, 'accepted', True)
        snapshot = Path(self.tmp.name) / 'accepted.jsonl'
        self.assertEqual(export_training_feedback(self.db, snapshot), 1)
        self.assertEqual(len(get_training_feedback_examples(self.db)), 1)
        review_feedback(self.db, fid, 1, 'rejected', True)
        self.assertEqual(get_training_feedback_examples(self.db), [])
        # Even an old export must not revive a correction after rejection.
        self.assertEqual(load_feedback_rows(self.db, 1, str(snapshot), 1), [])
        self.assertEqual(export_training_feedback(self.db, snapshot), 0)

    def test_source_overlap_does_not_grant_training_approval(self):
        advisor = SimpleNamespace(_ensure_rag_components=lambda **kw: None,
            embedder=SimpleNamespace(encode=lambda _: [[1]]),
            retriever=SimpleNamespace(retrieve=lambda *a, **kw: [{'text': 'soil test before sowing'}]))
        result = validate_feedback_with_local_sources(advisor, 'grow wheat', 'old', 'soil test before sowing', 'crop_guide')
        self.assertEqual(result['status'], 'source_matched')
        self.assertFalse(result['training_eligible'])
        result = validate_feedback_with_local_sources(advisor, 'weather', 'old', '', 'weather', rating='not_helpful')
        self.assertEqual(result['status'], 'needs_review')

    def test_training_split_excludes_dynamic_values_and_personal_data(self):
        for i in range(40):
            fid = self.add(question=f'How to grow crop variety {i}?')
            review_feedback(self.db, fid, 1, 'accepted', True)
        for topic in ['weather', 'price', 'crop_profitability']:
            fid = self.add(question=f'Latest {topic}?', topic=topic)
            review_feedback(self.db, fid, 1, 'accepted', True)
        fid = self.add(question='Email me at farmer@example.com')
        review_feedback(self.db, fid, 1, 'accepted', True)
        result = build_feedback_dataset(self.db)
        self.assertTrue(result['train'])
        self.assertTrue(result['eval'])
        self.assertEqual(sum(map(len, result.values())), 40)
        self.assertFalse({x['prompt'] for x in result['train']} & {x['prompt'] for x in result['eval']})
        self.assertEqual(result, build_feedback_dataset(self.db))

    def test_reviewer_must_supply_complete_safe_correction(self):
        fid = self.add(correction='')
        with self.assertRaises(ValueError):
            review_feedback(self.db, fid, 1, 'accepted', True)
        with self.assertRaises(ValueError):
            review_feedback(self.db, fid, 1, 'accepted', True, correction_text='Email farmer@example.com')
        review_feedback(self.db, fid, 1, 'accepted', True, correction_text='मिट्टी की जांच कराएं।')
        self.assertEqual(get_training_feedback_examples(self.db)[0]['correction_text'], 'मिट्टी की जांच कराएं।')


class ChatSuggestionsTests(unittest.TestCase):
    def test_component_submits_once_across_streamlit_reruns(self):
        from app.chat_controls import consume_submission
        state = {}
        value = {"id": "session-1", "text": " आज मौसम? "}
        self.assertEqual(consume_submission(value, state), "आज मौसम?")
        self.assertIsNone(consume_submission(value, state))
        self.assertEqual(consume_submission(dict(value, id="session-2"), state), "आज मौसम?")
        for bad in [None, "question", {}, {"id": "x", "text": " "}, {"id": "y", "text": "x" * 4001}]:
            self.assertIsNone(consume_submission(bad, state))


class FeedbackWidgetTests(unittest.TestCase):
    def test_weather_negative_rating_is_saved_without_requiring_correction(self):
        import ast
        from streamlit.testing.v1 import AppTest
        source = Path('streamlit_app.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'render_feedback_widget')
        widget = ast.get_source_segment(source, node)
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'db.sqlite')
            init_db(db)
            qid = create_query_log(db, dict(user_id=2, user_query='आज मौसम?', answer_text='आसमान साफ', topic='weather'))
            script = f'''
import streamlit as st
from types import SimpleNamespace
from pathlib import Path
from app.db import feedback_exists, save_feedback, export_training_feedback
from app.feedback import validate_feedback_with_local_sources
RAGAdvisor = object
cfg = SimpleNamespace(paths={{'sqlite_db': {db!r}}})
TRAINING_FEEDBACK_PATH = Path({str(Path(tmp) / 'feedback.jsonl')!r})
def current_user():
    return {{'id': 2}}
{widget}
render_feedback_widget({{'role':'assistant','query_log_id':{qid}, 'topic':'weather', 'user_query':'आज मौसम?', 'text':'आसमान साफ'}}, None)
'''
            at = AppTest.from_string(script).run()
            self.assertFalse(at.exception)
            at.radio[0].set_value('Not helpful').run()
            at.selectbox[0].set_value('हिंदी और अनुवाद').run()
            at.button[0].click().run()
            self.assertFalse(at.exception)
            with get_conn(db) as conn:
                row = conn.execute('SELECT * FROM answer_feedback').fetchone()
                self.assertEqual(row['rating'], 'not_helpful')
                self.assertEqual(row['validation_status'], 'needs_review')
                self.assertIn('हिंदी और अनुवाद', row['validation_notes'])
            self.assertEqual(get_training_feedback_examples(db), [])


if __name__ == '__main__':
    unittest.main()
