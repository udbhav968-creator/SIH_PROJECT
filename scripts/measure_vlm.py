"""
Score the vision-language second opinion (genai/vlm.py) on labelled photographs.

    set ANTHROPIC_API_KEY=...              (or run Ollama with a vision model and ROAD_SHIELD_OLLAMA_MODEL=llava)
    python -m scripts.measure_vlm --n 100  # ~100 API calls; costs money with Claude

Photographs: the held-out split's test set (training.train_cnn_head.capped_grouped_split), at most --n, in a
fixed shuffled order so every class appears. The VLM never saw any of them. The project's classifier is scored on
the same photographs for comparison, but the comparison is only fair where this machine's split is the one the
classifier was trained with (Colab); the report records the split size so that can be checked against
checkpoints/finetune_*_report.json.

Also reported: "adjudicated" = the classifier, except where its top probability is below --tau, where the VLM's
answer is used - what serving the VLM as a last cascade stage would score. Writes checkpoints/vlm_eval_report.json.
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def macro_f1(y, p):
    present = sorted(set(y))
    f = []
    for c in present:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return round(float(np.mean(f)), 4) if f else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.6)
    a = ap.parse_args(argv)

    from PIL import Image
    from genai import llm
    from genai.vlm import second_opinion
    from models.deep_vision_net import load_best_vision_model
    from training.train_cnn_head import capped_grouped_split

    p = llm.pick(require_vision=True)
    if p.name == "none":
        sys.exit("no vision-capable model: set ANTHROPIC_API_KEY, or run Ollama with a vision model")
    _, (_tr, _va, te) = capped_grouped_split(42, 1500)
    items = list(te)
    random.Random(7).shuffle(items)
    items = items[: a.n]
    clf, backend = load_best_vision_model(verbose=False)
    # every photograph counts: when the VLM does not answer, serving would fall back to the classifier, so the
    # adjudicated score uses the classifier there; the VLM-alone score is over the photographs it answered
    y, p_clf, p_adj, conf_clf, t0 = [], [], [], [], time.time()
    y_ans, p_vlm, agree, loose, failures = [], [], [], 0, 0
    for i, (path, label, _g) in enumerate(items):
        im = Image.open(path).convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode()
        pr = np.asarray(clf.predict_probabilities(np.asarray(im))).ravel()
        v = second_opinion(b64, provider=p)
        y.append(int(label))
        p_clf.append(int(pr.argmax()))
        conf_clf.append(float(pr.max()))
        if v.get("available"):
            y_ans.append(int(label))
            p_vlm.append(int(v["class_id"]))
            agree.append(int(v["class_id"]) == int(pr.argmax()))
            loose += bool(v.get("parsed_loosely"))
            p_adj.append(int(v["class_id"]) if pr.max() < a.tau else int(pr.argmax()))
        else:
            failures += 1
            p_adj.append(int(pr.argmax()))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)
    if not y_ans:
        sys.exit("the VLM answered no photograph")
    acc = lambda pp, yy=None: round(float(np.mean(np.array(pp) == np.array(yy if yy is not None else y))), 4)  # noqa: E731
    report = {
        "provider": p.name, "model": p.model, "photographs": len(y), "unanswered": failures,
        "answers_parsed_from_free_text": loose,
        "split_test_size_here": len(te),
        "vlm": {"accuracy": acc(p_vlm, y_ans), "macro_f1": macro_f1(y_ans, p_vlm), "answered": len(y_ans)},
        "classifier": {"backend": backend, "accuracy": acc(p_clf), "macro_f1": macro_f1(y, p_clf),
                       "fair_only_if_split_matches_training": True},
        "adjudicated": {"tau": a.tau, "accuracy": acc(p_adj), "macro_f1": macro_f1(y, p_adj),
                        "share_sent_to_vlm": round(float(np.mean(np.array(conf_clf) < a.tau)), 4)},
        "agreement_classifier_vs_vlm": round(float(np.mean(agree)), 4),
        "seconds": round(time.time() - t0, 1), "generated_unix": int(time.time()),
    }
    with open(os.path.join(ROOT, "checkpoints", "vlm_eval_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
