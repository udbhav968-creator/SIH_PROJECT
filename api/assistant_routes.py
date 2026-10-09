"""
The GenAI side of the API: the engineer's assistant (RAG), briefs for orders and defects, and the photo second
opinion. server.py calls handle_get / handle_post with what they need; they return True when they answered.

Cost control: a language model may be a paid API. Briefs and second opinions always need the operator key
when one is set; on a public deployment (ROAD_SHIELD_PUBLIC=1) questions need it too whenever a model is
configured. Questions are also under the server's per-client rate limit.
"""
import json
import os
import urllib.parse

from genai import llm, rag, report_writer, vlm

PROTECTED_POST = {"/api/v1/assistant/report", "/api/v1/assistant/second-opinion"}
KB = rag.KnowledgeBase()


def _models_text(ckpt):
    from models import served_report
    parts = []
    try:
        c = served_report.served_classifier_summary(ckpt) or {}
        if c:
            parts.append(f"The served image classifier is {c.get('label')}: held-out test accuracy "
                         f"{c.get('held_out_test_accuracy')}, macro-F1 {c.get('held_out_test_macro_f1')} on "
                         f"{c.get('held_out_test_images')} test images; on Indian roads (RDD2022) macro-F1 "
                         f"{(c.get('indian_roads') or {}).get('macro_f1')}.")
    except Exception:
        pass
    for name, label, keys in (("imu_model_selection.json", "IMU shock classifier", None),
                              ("ood_guard_report.json", "input guard", None),
                              ("ensemble_report.json", "ensemble", None)):
        p = os.path.join(ckpt, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if name == "imu_model_selection.json":
            ho = d.get("held_out_for_reporting") or {}
            parts.append(f"The IMU shock classifier served is {d.get('served')}; held-out accuracy "
                         + ", ".join(f"{k} {v.get('accuracy')}" for k, v in ho.items()) + ".")
        elif name == "ood_guard_report.json":
            t = d.get("test") or {}
            parts.append(f"The input guard has AUROC {(t.get('combined') or {}).get('auroc')} on held-out photographs; "
                         f"it refuses {(t.get('refusals') or {}).get('everyday_photos_refused_rate')} of everyday "
                         f"photographs and {(t.get('refusals') or {}).get('road_photos_refused_rate')} of road ones.")
        else:
            parts.append(f"Ensemble decision: {d.get('decision')}.")
    return " ".join(parts)


def live(dedup, works, priority_fn, ckpt):
    status = works.status_by_defect() if works is not None else {}
    # copies: the ledger's own records are never written to from here
    defects = [dict(d, repair=status.get(d.get("defect_id"))) for d in dedup.get_all_deduplicated_defects()]
    return rag.live_chunks(defects, priority_fn, works.list() if works is not None else [], _models_text(ckpt))


def handle_get(h, path, full_path, ctx):
    if path != "/api/v1/assistant/status":
        return False
    rep = None
    p = os.path.join(ctx["ckpt"], "rag_eval_report.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        rep = {"fresh_questions": {k: v for k, v in (r.get("fresh_questions") or {}).items() if k != "rows"},
               "method": r.get("method")}
    h._send_json(200, {"llm": llm.describe(), "vision_llm": llm.pick(require_vision=True).name,
                       "knowledge_chunks": len(KB.static_chunks()), "retrieval_eval": rep})
    return True


def handle_post(h, path, body, ctx, key_ok):
    if not path.startswith("/api/v1/assistant/"):
        return False
    body = body if isinstance(body, dict) else {}
    if path == "/api/v1/assistant/ask":
        # a paid model (Claude) costs money per question: when an operator key is set, only key holders may spend it;
        # on a public deployment every model needs the key
        prov = llm.pick().name
        if (prov == "anthropic" or (prov != "none" and os.environ.get("ROAD_SHIELD_PUBLIC") == "1")) \
                and not key_ok(h.headers):
            h._send_json(401, {"error": "the assistant uses a paid language model here, so it needs the operator key"})
            return True
        q = str(body.get("question") or "").strip()
        if not q:
            h._send_json(400, {"error": "question is required"})
            return True
        try:
            out = rag.answer(q, KB, live(ctx["dedup"], ctx["works"], ctx["priority"], ctx["ckpt"]))
        except ValueError as e:
            h._send_json(400, {"error": str(e)})
            return True
        h._send_json(200, out)
        return True
    if path == "/api/v1/assistant/report":
        oid, did = body.get("work_order_id"), body.get("defect_id")
        order = ctx["works"].get(str(oid)) if oid else None
        if oid and order is None:
            h._send_json(404, {"error": "no such work order"})
            return True
        if order:
            did = order.get("defect_id")      # a work order's brief is about its own defect, whatever else was sent
        defect = next((d for d in ctx["dedup"].get_all_deduplicated_defects() if d.get("defect_id") == did), None)
        if not order and not defect:
            h._send_json(404 if did else 400, {"error": "no such defect" if did else "work_order_id or defect_id is required"})
            return True
        pr = None
        if defect:
            try:
                pr = ctx["priority"](defect)
            except Exception:
                pr = None
        h._send_json(200, report_writer.write(order=order, defect=defect, priority=pr, kb=KB))
        return True
    if path == "/api/v1/assistant/second-opinion":
        b64 = body.get("image_base64")
        if not b64 or not isinstance(b64, str):
            h._send_json(400, {"error": "image_base64 is required"})
            return True
        if "," in b64[:100]:
            b64 = b64.split(",", 1)[1]
        h._send_json(200, vlm.second_opinion(b64, classifier=body.get("classifier")))
        return True
    return False
