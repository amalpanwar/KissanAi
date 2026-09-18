"""Inspect the configured model and optionally run a local inference smoke test."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Load the model and run a small grounded prompt')
    args = parser.parse_args()
    from app.config import load_config
    model = os.getenv('KISAANAI_COMPLEX_GENERATOR_MODEL') or os.getenv('KISAANAI_MEDIUM_GENERATOR_MODEL') or load_config(ROOT / 'configs/pipeline.yaml').generator_model
    report = {'model': model, 'python': sys.executable,
              'dependencies': {name: importlib.util.find_spec(name) is not None for name in ['torch','transformers','accelerate']},
              'inference_verified': False}
    if args.generate:
        from app.generator import LocalGenerator
        try:
            generator = LocalGenerator(model)
            report['response'] = generator.generate('सिर्फ दिए गए तथ्य से हिंदी में उत्तर दें: गेहूं का मूल्य 2500 रुपये प्रति क्विंटल है, रिपोर्ट दिनांक 18-09-2026। प्रश्न: गेहूं का भाव और तारीख क्या है?')
            report['inference_verified'] = bool(report['response'])
            report['note'] = 'A smoke test is not an agricultural accuracy benchmark.'
        except Exception as exc:
            report['error'] = type(exc).__name__
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not args.generate or report['inference_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
