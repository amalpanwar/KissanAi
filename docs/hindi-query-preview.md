# Hindi query preview and crop follow-ups

The composer offers a Hindi draft after a 900 ms typing pause. English and Hinglish drafts are sent server-side to Sarvam Mayura (`mayura:v1`, automatic source language, native Hindi output). It uses the existing `SARVAM_API_KEY` from Streamlit Secrets or the environment. Pure Hindi input does not call translation. The API key is never sent to the browser.

The original input stays editable. The separate Hindi preview is editable and becomes the posted question when sent. Users can disable the Hindi option to send their original wording. While translation is pending, submission waits or the user can disable it; failed/unconfigured translation leaves the original available for submission. Inputs over the model's 1,000-character limit fall back rather than being truncated. Requests have an eight-second timeout. Each paused draft can consume a translation request; no cross-user draft cache is used.

Draft events never enter chat history or invoke the advisor. Preview IDs and original-text matching prevent old results from replacing a newer draft. Edited previews survive unrelated reruns. Location changes create a new composer instance. Selected location names available in session context and numeric values are protected during translation; arbitrary proper names are not guaranteed, so users should review the preview. Translation does not guarantee correct intent or retrieval.

Variety, irrigation, fertilizer and similar follow-ups route to agronomy. An explicit crop overrides the remembered preferred crop; otherwise that context supplies the crop. If neither is present, the agent asks which crop. Crop document lookup runs before the general fallback, including supplementary PDFs when the primary guide is missing. If the guide cannot answer, research receives the resolved crop through the same agent message bus. This does not add missing source documents or invent locally suitable varieties.

Validation uses mocked Sarvam outputs and crop-document fixtures plus a real local Streamlit/Chrome composer integration test. Live Sarvam quality and the deployed rice-document contents were not verified.

API reference: https://docs.sarvam.ai/api-reference/text/translate-text
