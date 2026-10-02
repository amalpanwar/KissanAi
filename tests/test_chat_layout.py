"""Keep every live answer path above the composer, including early exits/errors."""
import ast
from pathlib import Path
import unittest


class ChatLayoutTests(unittest.TestCase):
    def test_entire_live_turn_renders_in_reserved_response_area(self):
        tree = ast.parse(Path('streamlit_app.py').read_text())
        area = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'response_area' for t in n.targets))
        composer = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'user_query' for t in n.targets)
                        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                        and n.value.func.id == 'render_chat_composer')
        turn = next(n for n in tree.body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Name) and n.test.id == 'user_query')
        self.assertLess(area.lineno, composer.lineno)
        self.assertEqual(len(turn.body), 1)
        self.assertIsInstance(turn.body[0], ast.With)
        self.assertEqual(turn.body[0].items[0].context_expr.id, 'response_area')
        self.assertTrue(any(isinstance(n, ast.Try) and n.handlers for n in ast.walk(turn.body[0])))


if __name__ == '__main__':
    unittest.main()
