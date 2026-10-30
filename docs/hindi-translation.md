# Local Hindi translation with IndicTrans2

English answer fragments are translated on the app host with AI4Bharat's
`ai4bharat/indictrans2-en-indic-dist-200M` checkpoint. Final-answer translation
does not call Sarvam or another translation API. Existing query-language
translation behavior is unchanged. Deterministic Hindi crop-guide answers
remain untouched.

## Model access and deployment

The model card requires accepting its access conditions on Hugging Face. Once
accepted, provide a Hugging Face read token as the top-level Streamlit secret
`HF_TOKEN` (or an environment variable for other hosts) so a fresh deployment
can download the model. Do not commit the token. The first English answer may
take longer while the model downloads and loads; subsequent chunks are cached
in process memory. The app host must have enough memory and disk for PyTorch,
Transformers, the model weights, and inference. The 200M distilled checkpoint
is smaller than the 1B checkpoint, but it is not a zero-resource model.

Example Streamlit secret:

```toml
HF_TOKEN = "your-hugging-face-read-token"
KISAANAI_HINDI_TRANSLATION = "1"
```

## Translation behavior

- The answer is split into short prose fragments; numeric values, units,
  formulas, links, markup, and explicit entity names are masked and restored.
- The app validates that masks are preserved and that output contains Hindi.
  If model load or translation fails, the original English answer is retained
  with a diagnostic status; source evidence and page references are unchanged.
- Successful text chunks are cached in process memory. This cache clears on app
  restart. Complete deterministic Hindi crop-guide responses skip translation.
- Set `KISAANAI_HINDI_TRANSLATION=0` to disable translation.

The checkpoint is MIT-licensed. Translation quality and deployment memory use
should be evaluated with representative agronomy answers before relying on it.
See the [official model card](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
and [IndicTrans2 documentation](https://github.com/AI4Bharat/IndicTrans2).
