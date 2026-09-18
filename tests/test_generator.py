import unittest
from unittest.mock import Mock
from app.generator import LocalGenerator


class GeneratorTests(unittest.TestCase):
    def test_instruct_chat_template_applied(self):
        generator = LocalGenerator.__new__(LocalGenerator)
        generator.tokenizer = Mock(chat_template="chatml")
        generator.tokenizer.apply_chat_template.return_value = "<user>hello<assistant>"
        generator.pipe = Mock(return_value=[{"generated_text": " उत्तर "}])
        self.assertEqual(generator.generate("hello"), "उत्तर")
        generator.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True)
        generator.pipe.assert_called_once_with("<user>hello<assistant>", return_full_text=False)
