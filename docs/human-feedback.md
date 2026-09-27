# Suggested questions and human feedback

The chat composer shows a dropdown of Hindi question suggestions on focus and
filters them as users type (Hindi, English, or Hinglish keywords). It covers weather,
mandi prices, cultivation and pesticides. There are no separate suggestion tabs,
shortcut buttons or draft form. Click a suggestion or use arrow keys and Enter to
fill the same input, edit it if needed, then send. Enter sends; Shift+Enter inserts
a newline; Escape closes suggestions. Only sending triggers the agent pipeline.

“My area”, “मेरे क्षेत्र”, “here” and similar phrases inherit the selected location.
A Doghat Rural selection keeps the full village/Baraut/Baghpat hierarchy and uses
its available coordinates. With only Baghpat selected, the district is used. Named
places still go through the existing scoped validation. Missing granular weather
coordinates continue to use a labelled regional fallback.

Signed-in users can rate any logged answer, including weather, and flag Hindi,
accuracy, missing information, or length/presentation. A negative rating does not
require the farmer to know the correct answer. Corrections remain optional except
when selecting **Provide correction**.

Administrators use **Feedback Review Queue** to see the question, original answer,
feedback area, retrieved evidence, and an editable replacement answer. Accept only
a complete, verified answer. For pesticide advice, a qualified agricultural reviewer
must check the crop, pest, registered product, dosage, units, and source date.
Keyword overlap is evidence for review, never automatic approval. Rejection removes
an example from future database retrieval and regenerated exports; stale export
files are no longer read as live feedback memory.

## Build a learning dataset

On the persistent server holding the app's SQLite database:

```sh
python scripts/prepare_feedback_training.py
# Or specify --db /path/to/kisaanai.db --output-dir /path/to/private/feedback
```

Only explicitly accepted, training-eligible corrections with a recorded reviewer
are selected. Empty examples and detected contact/personal data are excluded.
Weather, market prices, profitability and mixed answers are excluded: historical
live values should not become timeless training targets. Those ratings still help
reviewers find application problems. Remaining questions are deduplicated and
assigned deterministically to train/eval (approximately 80/20). Both sets must have
examples before tuning; a small dataset can produce an empty split.

```sh
python scripts/finetune_slm.py \
  --train-file data/processed/feedback_learning/train.jsonl \
  --eval-file data/processed/feedback_learning/eval.jsonl
```

This uses the existing optional model-training dependencies and compute; it does
not run during chat or automatically deploy an adapter. It reports held-out loss.
Also compare the candidate against the current model on unseen Hindi/Hinglish
questions, source fidelity, location names, dose/unit preservation and response
formatting. Loss alone does not establish factual or agricultural correctness.
Keep the current model until that comparison passes. Dataset files contain user
questions and should remain private; do not commit them. Use persistent storage
and back up the SQLite database—ephemeral hosting storage can lose feedback.

Ratings do not directly change model weights. Existing reviewed feedback memory
continues to aid matching; reviewed answer exports enable a separately evaluated
fine-tuning cycle. This change does not claim a measured model-quality improvement
or change the deployed model.
