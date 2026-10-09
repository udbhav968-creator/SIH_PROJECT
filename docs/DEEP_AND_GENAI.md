# Transformers, ensembles, RAG and generative AI

What was added on 9 Oct 2026, what has been measured, and what is waiting for a GPU run. Accuracy only rises when a
measurement says it does: every new model below replaces nothing until its own selection rule, run on data it was
not trained on, chooses it.

## 1. Vision transformers (train on Colab)

`training/train_finetune_cnn.py` now trains two transformers from `timm` beside the CNNs, with the same split,
augmentation, early stopping on validation and single test scoring:

| name | model | why |
|---|---|---|
| `deit_small` | DeiT-Small / ViT-S/16 (~22 M parameters, ~88 MB ONNX) | a pure vision transformer: global self-attention over 16x16 patches |
| `levit_256` | LeViT-256 (~19 M) | convolutional stem + attention stages, designed for fast CPU inference |

Only transformers pretrained with ImageNet mean/std are allowed (every network shares one preprocessing in serving;
the script refuses others), and only those under the 95 MB file limit. Larger ones (ViT-B, Swin-T, ConvNeXt-T) would
be trained and then excluded by that limit, so they are not listed.

## 2. Ensemble served as a cascade (a two-node DAG)

```
photo ──► fastest member ──(top probability ≥ τ)──────────────────► answer
                 │
                 └──(less sure)──► every member ──► mean of temperature-scaled probabilities ──► answer
```

Every train-only network (trained on the training split only; validation chose its best epoch but was never
trained on) leaves its ONNX file and its validation/test outputs in `checkpoints/ensemble_runs/`.
`training/select_ensemble.py` then decides **on validation only**:

- candidates: every subset of 2–4 networks (each file ≤ 95 MB); each member's temperature fitted on validation;
- choice: highest validation score = mean(accuracy, macro-F1), the score single networks are chosen by;
- served only if the **cascade** (what is actually served) beats the best single network's validation score by
  ≥ 0.01, and the members' CPU time with flip TTA is ≤ 80 ms;
- τ: the lowest threshold whose validation score is within 0.005 of the full ensemble, so most photos pay for
  one network;
- the test set is scored once, with a 95% paired-bootstrap interval of the gain (resampling test photographs).
  Choosing a subset, temperatures and τ on one validation set is still optimistic; the margin guards against it,
  and the test interval is the number to quote.

If chosen, `checkpoints/vision_ensemble.json` makes `models/ensemble_classifier.py` serve it, and the Models page and
the assistant report the cascade's numbers. Each result says which networks answered
(`frame_classification.cascade`). After a switch, rebuild the monitoring reference or rebaseline on the MLOps page,
because the confidence distribution changes. Otherwise the single network stays.

**Why this is not measured here:** the image split is a shuffle over the whole corpus, and the Colab corpus is larger
than this machine's (630 vs 252 test images). Locally, many "test" photos were in the served network's training data,
so any gain measured here would be inflated. The guard in `select_ensemble.py` refuses members scored on different
splits.

## 3. IMU transformer

`training/train_imu_deep.py` adds a small transformer encoder (50 ms patches, 2 pre-norm attention layers, attention
written out so ONNX export never meets a fused kernel) as a third deep candidate beside the 1-D CNN at two widths.
The rule is unchanged: a deep model is served only if it beats the RandomForest on time-blocked cross-validation.

## 4. RT-DETR detector candidate

`python -m training.train_rdd_detector --model rtdetr-l.pt --tag rtdetr` trains RT-DETR (a DETR-family transformer
detector, no NMS) as a candidate; `python -m scripts.select_rdd_detector --tag rtdetr` serves it only if its mAP@0.5 on
India validation photographs beats the served YOLO, and only if its ONNX file is ≤ 95 MB (GitHub's limit is 100 MB).
rtdetr-l exports at roughly 130 MB, so expect it to be measured and reported but not served. The ONNX decoder reads
RT-DETR's normalised-box output, and serving resizes to a square the way RT-DETR was validated.

## Running the GPU part

After `scripts/colab_train_all.sh` in the same Colab session:

```
!bash scripts/colab_train_deep.sh 2>&1 | tee -a logs/colab_deep.txt          # ~1.5-2.5 h on a T4
!RTDETR=1 bash scripts/colab_train_deep.sh ...                                # + RT-DETR (~1 h more)
```

It is resumable through Drive. A one-minute smoke run precedes each member, and it ends with
`road_shield_deep_outputs.zip`; copy it in like the other Colab outputs, then `python -m mlops register
vision_classifier` and `gate` / `promote`.

## 5. Engineer's assistant (RAG)

`/assistant` and `POST /api/v1/assistant/ask`. Each answer is drawn from:

- the project's documents (`docs/*.md`, README);
- road-maintenance notes written for this project (`genai/knowledge/road_maintenance.md`: distress types, PCI bands,
  repair methods, work-zone safety, monsoon readiness; they name IRC:82, IRC:SP:55, ASTM D6433 and MoRTH Section 500
  and say to read the standard itself for specifications);
- the live ledger: every defect with its priority and repair status, a summary by class, band and status, the work
  orders and the served models' measured accuracy.

**Retrieval:** hybrid of BM25 (with a short road-domain synonym table) and character 3–5-gram TF-IDF, fused by
reciprocal rank. Up to 2 of the 6 passages are reserved for the best-matching live records. No embedding model is
needed.

| retrieval (`python -m genai.eval_rag`) | recall@1 | recall@3 | recall@6 | MRR |
|---|---|---|---|---|
| 14 questions written after the retriever was built | 0.71–0.79 | 1.00 | 1.00 | 0.85–0.88 |
| 24 questions used while building it | 0.79 | 0.83 | 0.88 | 0.82 |

The 14 "fresh" questions were scored once before a final change (synonyms applied in both retrievers, then one
ambiguous synonym removed). The before figures are in `checkpoints/rag_eval_report.json`; with 14 questions, one
question is 7 points. The corpus is the live `docs/` folder, including this page, so editing a document moves the
numbers: the ranges are what was seen across the last edits of 9 Oct. The right passage was in the top 3 for all 14
every time. Re-run `python -m genai.eval_rag` for the current figures.

**Generation:** the model is told to answer only from the numbered sources, cite every statement, and copy numbers
exactly. The sources are wrapped as data, and text inside them is never an instruction. After it answers:

- every citation is checked to exist;
- every number in the answer is looked for in the sources it cites. Those checks cover '2,608' = '2608',
  '5 lakh' ≠ '5 crore' and 'Rs.75000'; numbers in the question do not count as support. Numbers that are not
  there are shown in red beside the answer.
- Numbers written as words are not checked.

Without a model, the best-matching sentences are quoted with their sources and labelled as quotations.

## 6. Briefs, and a second opinion on a photo

- **Engineer's brief** (button on the printable work order, `POST /api/v1/assistant/report`): written only from the
  order's sealed fields plus the repair guidance for its class (fixed: the repair method and site-safety sections
  for a pothole). A draft that states any number not in those fields or that guidance is discarded for a plain
  template, and the page says so.
- **Second opinion** (button on Inspection, `POST /api/v1/assistant/second-opinion`): a vision-language model picks
  one of the 7 classes, with a confidence and a reason. It is advisory and never changes a measurement.
  `python -m scripts.measure_vlm --n 100` scores it, and "classifier, but the VLM when the classifier is below 0.6",
  on labelled photos. A general model never trained on this project's photos, so any labelled set is fair for it.
  Not measured yet: it needs your API key and costs per call.

## Which language model

Picked automatically:

1. Claude, if `ANTHROPIC_API_KEY` is set. The model is `ROAD_SHIELD_LLM_MODEL`, default `claude-sonnet-5-5`.
2. Otherwise Ollama on this machine. `ollama pull llama3.2:3b`; use a vision model such as `llava` for second
   opinions.
3. Otherwise none.

Set `ROAD_SHIELD_LLM=anthropic|ollama|none` to force one. The key is read from the environment only. When an operator
key (`ROAD_SHIELD_API_KEY`) is set, briefs, second opinions and questions answered by Claude, a paid model, all need
it. On a public deployment every model needs it. Questions are also rate-limited per client.

## Not done

- No GPU run of the transformers, ensemble or RT-DETR yet: the code, tests and Colab script are ready.
- The VLM has no measurement until someone runs `measure_vlm` with a key.
- Retrieval is lexical: a question phrased with none of a document's words can still miss it. An embedding model
  would help and is the next step once the engine has room for one.
