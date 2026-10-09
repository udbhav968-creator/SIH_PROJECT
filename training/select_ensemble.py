"""
Ensemble of CNNs and vision transformers, served as a confidence cascade (a two-node DAG), chosen without
looking at the test set.

    python -m training.select_ensemble                 # after training/train_finetune_cnn.py (any --archs)

Input: <dir>/<arch>_logits.npz from every TRAIN-ONLY network (trained on the training split only; validation
chose its best epoch but was never trained on, so it can choose weights and thresholds without leaking), and
their ONNX files. scripts/colab_train_deep.sh writes them to checkpoints/ensemble_runs/ensemble/.

What is decided on validation only
    members      every subset of 2-4 networks (a fixed, exhaustive candidate set), combined by averaging
                 temperature-scaled probabilities; each member's temperature is fitted on validation NLL
    choice       the subset with the highest validation score = mean(accuracy, macro-F1), the same score
                 the single architectures are chosen by
    serve?       only if the CASCADE (what is served) beats the best single network's validation score by
                 >= MIN_GAIN, and the members' summed CPU time (flip TTA included) is <= MAX_CPU_MS; members over
                 MAX_MEMBER_MB are left out: a rule fixed here before any test number
    cascade      the fastest member answers alone when its top probability >= tau; otherwise every member
                 votes. tau is the LOWEST threshold whose validation score is within CASCADE_SLACK of the full
                 ensemble's, so most photographs pay for one network

What the test set is used for: scoring the chosen single network, the chosen ensemble and the cascade, once,
with a paired bootstrap (resampling test photographs) of the ensemble's gain. Nothing on test changes a choice.

Outputs: checkpoints/ensemble_report.json always; when the rule says serve, checkpoints/vision_ensemble.json
plus the member ONNX/sidecars copied to checkpoints/deep_vision_ens_<arch>.onnx, which
models/ensemble_classifier.py loads instead of the single network.
"""
import argparse
import glob
import itertools
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
CKPT = os.path.join(ROOT, "checkpoints")

MIN_GAIN = 0.01
MAX_CPU_MS = 80.0          # per photograph, flip TTA included (each member runs twice)
MAX_MEMBER_MB = 95.0       # every served file must fit GitHub's 100 MB limit
CASCADE_SLACK = 0.005
MAX_MEMBERS = 4


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 60))
    return float(min(grid, key=lambda t: nll(softmax(logits / t), y)))


def macro(y, p, n_classes):
    present = sorted(set(y.tolist()))
    f1s = []
    for c in present:
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def score(y, probs, n_classes):
    pred = probs.argmax(1)
    acc = float(np.mean(pred == y))
    f1 = macro(y, pred, n_classes)
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "score": round(0.5 * (acc + f1), 4)}


def cascade_probs(first, full, tau):
    use_first = first.max(axis=1) >= tau
    return np.where(use_first[:, None], first, full), float(1 - use_first.mean())


def bootstrap_gain(y, groups, pa, pb, n_classes, reps=2000, seed=0):
    """95% interval of score(b) - score(a), resampling test photographs (crops of one photo move together)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(reps):
        pick = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        diffs.append(score(y[pick], pb[pick], n_classes)["score"] - score(y[pick], pa[pick], n_classes)["score"])
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"mean": round(float(np.mean(diffs)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "resampled": "test photographs", "reps": reps}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(CKPT, "ensemble_runs", "ensemble"))
    ap.add_argument("--out", default=CKPT)
    a = ap.parse_args(argv)

    members = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*_logits.npz"))):
        arch = os.path.basename(f)[: -len("_logits.npz")]
        side = os.path.join(a.dir, f"deep_vision_{arch}.json")
        if not os.path.exists(side) or not os.path.exists(os.path.join(a.dir, f"deep_vision_{arch}.onnx")):
            continue
        z = np.load(f, allow_pickle=False)
        members[arch] = {k: z[k] for k in z.files}
        with open(side, encoding="utf-8") as fh:
            onnx = json.load(fh).get("onnx") or {}
        # the measured time is one forward pass; serving runs the image and its mirror (flip TTA)
        members[arch]["cpu_ms"] = 2 * float(onnx.get("onnx_cpu_ms_per_image") or 999)
        if float(onnx.get("onnx_size_mb") or 0) > MAX_MEMBER_MB:
            print(f"  {arch}: ONNX {onnx.get('onnx_size_mb')} MB > {MAX_MEMBER_MB} MB - left out")
            del members[arch]
    if len(members) < 2:
        sys.exit(f"need at least two train-only networks in {a.dir}; found {list(members)}")
    names = sorted(members)
    ref = members[names[0]]
    for n in names[1:]:                               # every member must have been scored on the same photographs
        for k in ("val_y", "test_y", "val_groups", "test_groups"):
            if not np.array_equal(members[n][k], ref[k]):
                sys.exit(f"{n} was not scored on the same split as {names[0]} ({k} differs); retrain on one corpus")
    yv, yt = ref["val_y"].astype(int), ref["test_y"].astype(int)
    n_classes = int(max(members[names[0]]["val_logits"].shape[1], 1))

    for n in names:
        m = members[n]
        m["T"] = fit_temperature(m["val_logits"], yv)
        m["pv"], m["pt"] = softmax(m["val_logits"] / m["T"]), softmax(m["test_logits"] / m["T"])
        m["val"] = score(yv, m["pv"], n_classes)

    single = max(names, key=lambda n: (members[n]["val"]["score"], -members[n]["cpu_ms"]))
    candidates = []
    for k in range(2, min(MAX_MEMBERS, len(names)) + 1):
        for combo in itertools.combinations(names, k):
            pv = np.mean([members[n]["pv"] for n in combo], axis=0)
            candidates.append({"members": list(combo), "val": score(yv, pv, n_classes),
                               "cpu_ms": round(sum(members[n]["cpu_ms"] for n in combo), 1)})
    affordable = [c for c in candidates if c["cpu_ms"] <= MAX_CPU_MS] or candidates
    best = max(affordable, key=lambda c: (c["val"]["score"], -c["cpu_ms"]))

    # cascade: fastest member first
    first = min(best["members"], key=lambda n: members[n]["cpu_ms"])
    full_v = np.mean([members[n]["pv"] for n in best["members"]], axis=0)
    full_t = np.mean([members[n]["pt"] for n in best["members"]], axis=0)
    target = best["val"]["score"] - CASCADE_SLACK
    tau, esc_v = 1.01, 1.0
    for t in np.round(np.arange(0.30, 1.0001, 0.01), 2):
        cp, esc = cascade_probs(members[first]["pv"], full_v, t)
        if score(yv, cp, n_classes)["score"] >= target:
            tau, esc_v = float(t), esc
            break
    casc_t, esc_t = cascade_probs(members[first]["pt"], full_t, tau)
    casc_val = score(yv, cascade_probs(members[first]["pv"], full_v, tau)[0], n_classes)
    expected_ms = round(members[first]["cpu_ms"] + esc_v * (best["cpu_ms"] - members[first]["cpu_ms"]), 1)
    # what is served is the CASCADE, so the cascade (not the full ensemble) must clear the margin
    gain_val = round(casc_val["score"] - members[single]["val"]["score"], 4)
    serve = gain_val >= MIN_GAIN and best["cpu_ms"] <= MAX_CPU_MS
    served_summary = {}
    try:
        with open(os.path.join(a.out, "finetune_summary.json"), encoding="utf-8") as fh:
            fs = json.load(fh)
        served_summary = {"network": fs.get("served"), "test": fs.get("served_test"),
                          "note": "the currently served single network (refit on train+val, so it has no validation "
                                  "score); its test numbers are on the same test photographs only if this run used "
                                  "the same corpus - compare split sizes"}
    except Exception:
        pass

    tg = ref["test_groups"]
    report = {
        "generated_unix": int(time.time()),
        "rule": {"min_validation_gain": MIN_GAIN, "max_cpu_ms": MAX_CPU_MS, "cascade_slack": CASCADE_SLACK,
                 "candidates": "every subset of 2-%d train-only networks, mean of temperature-scaled probabilities" % MAX_MEMBERS,
                 "selection": "validation score = mean(accuracy, macro-F1); test scored once, never used to choose"},
        "members": {n: {"temperature": round(members[n]["T"], 3), "validation": members[n]["val"],
                        "test": score(yt, members[n]["pt"], n_classes), "cpu_ms": members[n]["cpu_ms"]} for n in names},
        "best_single": single,
        "best_ensemble": {**best, "test": score(yt, full_t, n_classes)},
        "cascade": {"first": first, "tau": tau, "validation_escalation_rate": round(esc_v, 4),
                    "validation": casc_val, "validation_gain_over_best_single": gain_val,
                    "test": score(yt, casc_t, n_classes), "test_escalation_rate": round(esc_t, 4),
                    "expected_cpu_ms": expected_ms},
        "served_single_network_for_reference": served_summary,
        "caveats": ["validation chose each member's best epoch (early stopping) and, here, the subset, the "
                    "temperatures and tau; the margin guards against, but does not remove, that optimism - the "
                    "test bootstrap is the number to quote",
                    "after switching the served classifier, rebuild the monitoring reference (python -m "
                    "training.train_ood_guard) or rebaseline on the MLOps page: the confidence histogram changes"],
        "test_gain_of_ensemble_over_single": bootstrap_gain(yt, tg, members[single]["pt"], full_t, n_classes),
        "test_gain_of_cascade_over_single": bootstrap_gain(yt, tg, members[single]["pt"], casc_t, n_classes),
        "top_candidates_by_validation": sorted(candidates, key=lambda c: -c["val"]["score"])[:8],
        "decision": "serve the cascade" if serve else
                    f"keep the single network (cascade's validation gain {gain_val} < {MIN_GAIN} or CPU "
                    f"{best['cpu_ms']} ms > {MAX_CPU_MS})",
        "test_images": int(len(yt)), "validation_images": int(len(yv)),
    }
    with open(os.path.join(a.out, "ensemble_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    sel = os.path.join(a.out, "vision_ensemble.json")
    if serve:
        for n in best["members"]:
            for ext in ("onnx", "json"):
                shutil.copyfile(os.path.join(a.dir, f"deep_vision_{n}.{ext}"),
                                os.path.join(a.out, f"deep_vision_ens_{n}.{ext}"))
        with open(sel, "w") as fh:
            json.dump({"served": True, "members": best["members"], "first": first, "tau": tau,
                       "temperatures": {n: round(members[n]["T"], 4) for n in best["members"]},
                       "report": "checkpoints/ensemble_report.json"}, fh, indent=1)
    elif os.path.exists(sel):
        os.remove(sel)
    print(json.dumps({k: report[k] for k in ("best_single", "best_ensemble", "cascade", "decision",
                                             "test_gain_of_ensemble_over_single")}, indent=1))
    return report


if __name__ == "__main__":
    main()
