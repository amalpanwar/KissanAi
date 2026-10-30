# Hindi answer translation

Final English answer prose is translated with the app's configured local instruction model (`Qwen/Qwen2.5-0.5B-Instruct` by default). Final-answer translation does not call Sarvam or send answer text to a translation provider. Existing retrieval, ranking, citations, and English answer synthesis remain separate.

## Runtime behavior

- The configured generator model is loaded once and reused for answer translation. It uses the same local model configuration as the crop guide.
- English fragments are translated into Devanagari Hindi. Existing Hindi text, references, and answer structure stay in the app.
- Technical names, quantities, units, formulas, and URLs are masked before translation and restored locally. A translated fragment is accepted only when protected values remain unchanged, no new numbers appear, and Hindi text is present.
- Validated chunks stream in source order. If one chunk fails, its original English is shown alongside successful Hindi chunks. If all chunks fail, the full source-grounded English answer remains visible.
- Set `KISAANAI_HINDI_TRANSLATION=0` to disable translation.
- Model weights may need to be available in the local Hugging Face cache. Set `KISAANAI_HF_ONLINE=1` on the host to allow downloading model weights. This downloads the model, not farmer answer text.

## Limits

The local Qwen model is small and may produce awkward or incomplete Hindi. Numeric checks do not prove that the meaning, negation, timing, or regional applicability was preserved. Inspect representative answers, especially pesticide and fertilizer guidance. If the local model is unavailable, the answer remains visible in English and the translation status reports the failure.

The query translation path is separate; this document describes only final-answer translation.
