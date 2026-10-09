"""
A vision-language model's second opinion on a road photograph.

Used when the classifier is unsure (the Inspect page offers it; nothing calls it automatically): the photograph
goes to a vision-capable model (Claude, or a vision model under Ollama such as llava) with the seven classes and
a request for one class, a confidence and a one-line reason, as JSON. The answer is shown beside the classifier's
and never replaces a measurement, an area, a priority or a work order.

How good is it? scripts/measure_vlm.py scores it on labelled photographs. A general model was never trained on
this project's photographs, so any labelled set is a fair test for it - unlike for the project's own networks.
"""
import json
import re

from genai import llm

CLASSES = ["Normal Road / Sound Pavement", "Crack (Longitudinal / Transverse / Alligator)", "Pothole Cavity",
           "Waterlogging / Flooding Hazard", "Missing Zebra Crossing", "Missing Road Divider", "Damaged Traffic Sign"]
KEYWORDS = [(r"\bpotholes?\b", 2), (r"\bcrack(s|ed|ing)?\b", 1), (r"\bwater(logg\w*)?\b|\bflood\w*", 3),
            (r"\bzebra\b", 4), (r"\bdivider\b|\bmedian\b", 5), (r"\b(traffic|road) signs?\b", 6),
            (r"\bnormal\b|\bsound\b", 0)]

SYSTEM = ("You inspect road photographs for an Indian municipal road-maintenance system. Look only at the road and "
          "its markings, dividers and signs. Reply with one JSON object and nothing else.")
PROMPT = ("Which ONE of these best describes the most important condition in the photograph?\n"
          + "\n".join(f"{i}. {c}" for i, c in enumerate(CLASSES))
          + '\n\nReply exactly as JSON: {"class_index": <0-6>, "confidence": <0.0-1.0>, "reason": "<at most 25 words>"}.'
          " If it is not a photograph of a road, use class_index 0 and say so in reason.")


def parse(text):
    """The model's JSON, tolerant of code fences or a sentence around it; None when unusable."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except ValueError:
            d = None
        if isinstance(d, dict):
            # a JSON answer is taken or refused as JSON: an out-of-range or non-integer class is unusable, and
            # its free text is not then mined for keywords ('crossing guard' is not a zebra crossing)
            i = d.get("class_index")
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(CLASSES):
                return None
            conf = d.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) and 0 <= conf <= 1 else None
            return {"class_id": i, "class_name": CLASSES[i], "confidence": conf, "reason": str(d.get("reason") or "")[:240]}
    low = (text or "").lower()
    found = {i for pat, i in KEYWORDS if re.search(pat, low)}
    if len(found) == 1:                       # only an unambiguous free-text answer is used, and flagged as loose
        i = found.pop()
        return {"class_id": i, "class_name": CLASSES[i], "confidence": None,
                "reason": "parsed from free text: " + low[:160], "parsed_loosely": True}
    return None


def second_opinion(image_b64, classifier=None, provider=None):
    """image_b64: JPEG/PNG bytes as base64 (no data: prefix). classifier: the pipeline's frame classification."""
    p = provider or llm.pick(require_vision=True)
    if p.name == "none" or not getattr(p, "vision", False):
        return {"available": False, "reason": "no vision-capable language model is configured "
                                              "(ANTHROPIC_API_KEY, or an Ollama vision model such as llava)"}
    try:
        r = p.complete(SYSTEM, PROMPT, images=[image_b64], max_tokens=200, temperature=0.0)
    except llm.LLMError as e:
        return {"available": False, "reason": str(e)}
    got = parse(r["text"])
    if got is None:
        return {"available": False, "reason": "the model's answer could not be read", "raw": r["text"][:300],
                "provider": r["provider"]}
    out = {"available": True, **got, "provider": r["provider"], "model": r.get("model"), "ms": r.get("ms"),
           "role": "advisory second opinion; it does not change any measurement"}
    if isinstance(classifier, dict) and classifier.get("class_name"):
        out["agrees_with_classifier"] = classifier["class_name"] == got["class_name"]
    return out
