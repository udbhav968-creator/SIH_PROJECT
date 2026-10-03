"""Build the served-model-dependent report edits from the checkpoints on disk."""
import json
import os


def load(ck, name):
    p = os.path.join(ck, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return f"{100 * x:.1f}%"


def f2(x):
    return f"{x:.2f}"


def build(ck, tests_line, bench=None):
    head = load(ck, "cnn_head_mobilenetv2_report.json")
    sel = load(ck, "vision_model_selection.json") or {}
    ft = load(ck, "finetune_summary.json")
    india = load(ck, "indian_roads_eval_report.json") or {}
    imu_sel = load(ck, "imu_model_selection.json")
    deep = sel.get("served") == "deep_cnn" and ft
    base = head["handcrafted_baseline"]

    if deep:
        rep = load(ck, f"finetune_{ft['served']}_report.json")
        t = rep["test"]
        acc, mf1, n_img, n_ph = t["accuracy"], t["macro_f1"], t["images"], rep["split"]["test_photographs"]
        pcs = t["per_class"]
        per = [(pcs[k]["precision"], pcs[k]["recall"], pcs[k]["f1"], pcs[k]["support"]) for k in head["class_names"]]
        macro_p = sum(p[0] for p in per) / 7
        macro_r = sum(p[1] for p in per) / 7
        arch = ft["chosen_arch"]
        arch_h = {"efficientnet_b0": "EfficientNet-B0", "efficientnet_b2": "EfficientNet-B2", "resnet50": "ResNet-50",
                  "convnext_tiny": "ConvNeXt-Tiny", "mobilenet_v3_large": "MobileNetV3-Large"}.get(arch, arch)
        ms = rep["onnx"]["onnx_cpu_ms_per_image"]
        ind = rep.get("indian_roads_rdd2022_test") or {}
        model_phrase = f"an ImageNet-pretrained {arch_h} fine-tuned end to end on our road photographs, run on CPU via ONNX Runtime"
    else:
        acc, mf1 = head["held_out_test_accuracy"], head["held_out_test_macro_f1"]
        n_img, n_ph = head["held_out_test_images"], head["held_out_test_photographs"]
        pr = head["per_class_report"]
        per = [(pr[k]["precision"], pr[k]["recall"], pr[k]["f1-score"], int(pr[k]["support"])) for k in head["class_names"]]
        macro_p, macro_r = pr["macro avg"]["precision"], pr["macro avg"]["recall"]
        arch_h = "MobileNetV2"
        ms = 30
        ind = india.get("after") or india.get("served") or india.get("frozen_head") or {}
        model_phrase = "a frozen ImageNet MobileNetV2 feature backbone with a trained classifier head, run on CPU via ONNX Runtime"

    big3 = sum(p[3] for p in per[:3])
    f1_big = [p[2] for p in per[:3]]
    rare_n = sorted({p[3] for p in per[3:]})
    rows = {}
    keys = ["Normal Road / Sound Pavement", "Crack (Longitudinal", "Severe Cavity / Pothole",
            "Waterlogging / Flooding Hazard", "Missing Zebra Crossing", "Missing Road Divider", "Damaged Traffic Sign"]
    for k, (p, r, f, n) in zip(keys, per):
        rows[k] = [f2(p), f2(r), f2(f), str(n)]
    rows["Serving Model (Overall"] = [f"{macro_p:.3f}", f"{macro_r:.3f}", f"Macro: {mf1:.3f}", f"Accuracy: {pct(acc)} ({n_img})"]

    ind_txt = (f" On {ind['images']} crops from held-out RDD2022 India photographs (split by photograph; no training run reads them), it scored "
               f"{pct(ind['accuracy'])} accuracy (macro-F1 {ind['macro_f1']:.3f}).") if ind else ""

    E = []
    E.append(("our classifier scored 88.8% accuracy (macro-F1 0.747) on 502 images from photographs it never saw, the hand-crafted baseline scored 79.9% on the same split, and all 140 automated tests pass end-to-end",
              f"our classifier scored {pct(acc)} accuracy (macro-F1 {mf1:.3f}) on {n_img} images from {n_ph} photographs it never saw, the hand-crafted baseline scored {pct(base['accuracy'])} on the same split, and the code has {tests_line}"
              + (f". On {ind['images']} crops from held-out RDD2022 India photographs (split by photograph; no training run reads them), the classifier scored {pct(ind['accuracy'])} accuracy (macro-F1 {ind['macro_f1']:.3f})" if ind else "")))
    E.append(("It uses a ResNet-50 feature backbone running on CPU via ONNX Runtime to spot 9 classes of road defects,",
              f"It uses {model_phrase} to sort each region into 7 road-condition classes,"))
    if deep:
        E.append(("frozen ImageNet CNN embeddings (MobileNetV2 served, ResNet-50 as the larger configuration); 88.8% held-out accuracy",
                  f"an ImageNet {arch_h} fine-tuned end to end; {pct(acc)} held-out accuracy"))
        E.append(("Extracts 1000-d embeddings with a frozen ImageNet CNN (MobileNetV2 served; ResNet-50 as the larger configuration), averaging each image with its mirror; a soft-voting ensemble (SVC, logistic regression, MLP) chosen by grouped cross-validation predicts across 7 distress classes.",
                  f"An ImageNet-pretrained {arch_h}, fine-tuned end to end on the road corpus (chosen among {len(ft['archs'])} architectures on validation data, then refit on train+val), predicts across 7 distress classes, averaging each crop with its mirror. The frozen MobileNetV2 + scikit-learn head is kept as the fallback."))
        E.append(("ImageNet MobileNetV2 Embeddings (ONNX Runtime); ResNet-50 as the larger configuration",
                  f"ImageNet {arch_h} fine-tuned end to end (ONNX Runtime)"))
        E.append(("Pre-trained ImageNet features transfer well to road texture. On the identical split the CNN embeddings beat HOG/LBP (88.8% vs 79.9% accuracy), at about 30 ms per image on CPU.",
                  f"On the identical split the fine-tuned network scored {pct(acc)} (macro-F1 {mf1:.3f}), the frozen MobileNetV2 head {pct(head['held_out_test_accuracy'])} (macro-F1 {head['held_out_test_macro_f1']:.3f}) and HOG/LBP {pct(base['accuracy'])}; it was selected by a rule fixed before test scores were read (it had to beat the head on both accuracy and macro-F1 on non-test data). About {ms:.0f} ms per image on CPU."))
        E.append(("Soft-Voting Ensemble (class-balanced SVC + logistic regression + MLP) on mirror-averaged embeddings",
                  f"Linear classifier inside the fine-tuned {arch_h} (label smoothing, square-root class-balanced sampling)"))
        E.append(("Each head alone; logistic regression at two regularisation strengths; SVC-RBF at three",
                  "Frozen MobileNetV2 + soft-voting head; " + ", ".join(k for k in ft["archs"] if k != arch) + " fine-tuned the same way"))
        hv = ft["archs"][arch]["val"]
        E.append(("Selected by 5-fold cross-validation grouped by source photograph within the training split (CV accuracy 84.9%, macro-F1 0.769, best of 13 candidates); the held-out test set was used once, to score the selected model.",
                  f"Selected on the validation split (accuracy {pct(hv['accuracy'])}, macro-F1 {hv['macro_f1']:.3f}); the held-out test set was scored once per architecture and never used to choose."))
    else:
        cv = head["heads_compared"][head["head"] + "|" + head["features"]]
        E.append(("frozen ImageNet CNN embeddings (MobileNetV2 served, ResNet-50 as the larger configuration); 88.8% held-out accuracy",
                  f"frozen ImageNet MobileNetV2 embeddings; {pct(acc)} held-out accuracy"))
        E.append(("Extracts 1000-d embeddings with a frozen ImageNet CNN (MobileNetV2 served; ResNet-50 as the larger configuration), averaging each image with its mirror; a soft-voting ensemble (SVC, logistic regression, MLP) chosen by grouped cross-validation predicts across 7 distress classes.",
                  f"Extracts 1000-d embeddings with a frozen ImageNet MobileNetV2, averaging each image with its mirror; a class-balanced {head['head'].split('_C')[0].upper()} (C={head['head'].split('_C')[1]})" + " head chosen by grouped cross-validation predicts across 7 distress classes."))
        E.append(("ImageNet MobileNetV2 Embeddings (ONNX Runtime); ResNet-50 as the larger configuration",
                  "ImageNet MobileNetV2 Embeddings (ONNX Runtime)"))
        E.append(("Pre-trained ImageNet features transfer well to road texture. On the identical split the CNN embeddings beat HOG/LBP (88.8% vs 79.9% accuracy), at about 30 ms per image on CPU.",
                  f"Pre-trained ImageNet features transfer well to road texture. On the identical split the CNN embeddings beat HOG/LBP ({pct(acc)} vs {pct(base['accuracy'])} accuracy), at about 30 ms per image on CPU."))
        E.append(("Soft-Voting Ensemble (class-balanced SVC + logistic regression + MLP) on mirror-averaged embeddings",
                  f"Class-balanced SVC-RBF (C=3) on mirror-averaged embeddings"))
        E.append(("Selected by 5-fold cross-validation grouped by source photograph within the training split (CV accuracy 84.9%, macro-F1 0.769, best of 13 candidates); the held-out test set was used once, to score the selected model.",
                  f"Selected by 5-fold cross-validation grouped by source photograph within the training split (CV accuracy {pct(cv['cv_accuracy'])}, macro-F1 {cv['cv_macro_f1']:.3f}, best of {len(head['heads_compared'])} candidates); the held-out test set was used once, to score the selected model."))

    E.append(("472 of the 502", f"{big3} of the {n_img}"))
    E.append(("(0.87 to 0.91)", f"({min(f1_big):.2f} to {max(f1_big):.2f})"))
    E.append(("only 7 or 8 test photos each", f"only {rare_n[0]} or {rare_n[-1]} test photos each" if len(rare_n) > 1 else f"only {rare_n[0]} test photos each"))
    E.append(("The repository's automated suite of 140 tests passes in full (1 skipped by design), covering every model, the inference pipeline, the REST API end to end, and the serverless entrypoint.",
              f"The repository's automated suite: {tests_line}. It covers every model, the inference pipeline, the REST API end to end, the serverless entrypoint, and regression tests for every hard-coded value removed in the October 2026 audit."))
    E.append(("852 windows (1.0 s) from 10 real drive logs, 205,491 samples", "852 windows (1.0 s): 688 train, 164 held out, from 10 real drive logs (205,501 rows)"))
    E.append(("Smooth Asphalt, Expansion Joint, Rumble Strip / Speed Breaker, Pothole Impact", "Smooth Asphalt, Unmarked Speed Breaker / Bump, Marked Speed Breaker, Pothole Impact"))
    E.append(("Real field recordings from Indian road drives, but a small set (10 logs, one project) rather than a bus fleet; split by time block between training and validation.",
              "Real field recordings from Indian road drives (car, not bus) published by one GitHub project; a small set of 10 logs, split by time block (first 80% / last 20% of each log)."))
    rdd = (india or {}).get("dataset") or {}
    if rdd:
        tc, ec = rdd.get("training_crops", {}), rdd.get("eval_crops", {})
        E.append(("RDD2022 (India Subset) Arya et al. — planned", "RDD2022 India (Arya et al., CRDDC 2022)"))
        E.append(("Not yet ingested", f"Training: {tc.get('crack')} crack, {tc.get('pothole')} pothole, {tc.get('normal')} normal-road crops from {rdd.get('training_photographs')} photographs; Indian test set: {sum(ec.values())} crops from {rdd.get('eval_photographs')} photographs"))
        E.append(("Identified as the Indian validation set; ingestion is scheduled for Milestone 3 (Section 11). The repository folder of this name currently holds 28 distinct images, 27 of them copies of DNIT/Kaggle photographs, so no Indian validation result is claimed here.",
                  "Split by photograph: train+valid photographs give training crops (prefixed rddin_), test photographs go only to datasets/_eval_rdd2022_india, which no training run reads. (The folder named 01_rdd2022_india holds DNIT copies, not RDD data, and is not used.)"))
    E.append(("Quantize our CNN backbones (MobileNetV2 and ResNet-50)", "Time the INT8 ONNX models already in checkpoints/int8 and quantize the served classifier"))
    if bench:
        lat_s = bench["mean_ms"] / 1000.0
        E.append(("classifying an image in about 30 ms without needing an expensive, power-hungry GPU; the full pipeline including pixel segmentation takes 1.7–2.4 s on a single-core machine.",
                  f"classifying an image region in about {ms:.0f} ms without needing a GPU; the full per-photograph pipeline (segmentation, semantic gate, classification, costing) averaged {lat_s:.1f} s on {bench['where']}."))
        E.append(("with about 30 ms classification latency per image (the full pipeline including pixel segmentation takes 1.7–2.4 s on a single-core machine),",
                  f"- {bench['n']} distinct photographs drawn from the dataset folders (an end-to-end run, not a held-out accuracy test) - with a mean of {lat_s:.1f} s per photograph on {bench['where']},"))
        tail = (f" estimated {bench['tonnes']:.3f} t of bituminous mix in total" if bench.get("tonnes") is not None else
                " priced repairs only where the area was plausible for a single patch")
        tail += f" (total {bench['cost']:,.0f} INR)"
        tail += (f", and {bench['seal_ok']}/{bench['seal_n']} sealed work orders verified." if bench.get("seal_n") else
                 "; the earlier '1.492 t' and '100% seal verification' figures are not supported by any file in the repository and are withdrawn.")
        E.append((" calculated 1.492 tons of bitumen needed, and achieved 100% cryptographic seal verification.", tail))
    return {"text_edits": E, "rows": rows,
            "summary": dict(acc=acc, mf1=mf1, n_img=n_img, n_ph=n_ph, deep=bool(deep), arch=arch_h, ind=ind,
                            base=base, ms=ms, imu=imu_sel)}
