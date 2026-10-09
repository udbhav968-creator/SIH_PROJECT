"""GenAI: retrieval, grounding checks, the LLM providers' wire format, the report writer and the VLM parser."""
import base64
import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from genai import llm, rag, report_writer, vlm  # noqa: E402


class FakeProvider:
    name, model, vision = "fake", "fake-1", True

    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, system, user, images=None, **kw):
        self.calls.append({"system": system, "user": user, "images": images})
        return {"text": self.text, "provider": self.name, "model": self.model, "ms": 1}


class _Server:
    """A local HTTP server that records requests and answers like the real API would."""

    def __init__(self, reply):
        seen = self.seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                seen.append(("GET", self.path, dict(self.headers), None))
                self._send(reply("GET", self.path, None))

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                seen.append(("POST", self.path, dict(self.headers), body))
                self._send(reply("POST", self.path, body))

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _png_b64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (90, 90, 90)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class RetrievalTest(unittest.TestCase):
    def test_chunks_keep_their_document_title(self):
        md = "# Road notes\n\nintro text here\n\n## Repairing a pothole\n\ncut and patch is the planned repair.\n"
        ch = rag.chunk_markdown(md, "x.md")
        self.assertIn("Road notes > Repairing a pothole", [c["title"] for c in ch])

    def test_hybrid_finds_the_repair_notes_and_word_forms(self):
        kb = rag.KnowledgeBase()
        hits = rag.hybrid_search(kb.static_chunks(), "how do we fix a pothole properly", k=rag.TOP_K)   # what the model sees
        self.assertTrue(any(h["source"].endswith("road_maintenance.md") and "Repairing" in h["title"] for h in hits),
                        [h["title"] for h in hits])
        docs = [{"source": "a", "title": "Active learning", "text": "Operators label the queued photographs."},
                {"source": "b", "title": "Hosting", "text": "Keep the engine online with a free server."}]
        self.assertEqual(rag.hybrid_search(docs, "labelling queue", k=1)[0]["source"], "a")

    def test_retrieval_evaluation_meets_its_floor(self):
        from genai.eval_rag import QA_FRESH, evaluate
        rep = evaluate(qa=QA_FRESH)
        self.assertGreaterEqual(rep["recall_at_3"], 0.8, [r for r in rep["rows"] if not r["rank"] or r["rank"] > 3])


class GroundingTest(unittest.TestCase):
    SOURCES = [{"text": "Defect DEF-1 has PCI 42 and area 1.85 m2.", "title": "Defect DEF-1"},
               {"text": "Priority P1 means repair within 24 h.", "title": "Bands"}]

    def test_numbers_and_citations_are_checked(self):
        ok = rag.check_grounding("DEF-1 has PCI 42 [1] and must be repaired within 24 h [2].", self.SOURCES)
        self.assertTrue(ok["grounded"], ok)
        bad = rag.check_grounding("DEF-1 has PCI 55 [1]; cost INR 12,500 [3].", self.SOURCES)
        self.assertEqual(bad["unsupported_numbers"], [55.0, 12500.0])
        self.assertEqual(bad["invalid_citations"], [3])
        # loopholes the review found: glued numbers, small numbers, scale words, the question's own figure,
        # and numbers that only an UNcited source contains
        for text, q in (("PCI 42 [1], cost INR99999 [1].", ""), ("Send 3 crews [1].", ""),
                        ("Budget 5 crore [1].", ""), ("Yes, 999999 [1].", "Is it 999999?"),
                        ("Repair within 24 h [1].", "")):
            src = self.SOURCES if "24" not in text else [self.SOURCES[0], self.SOURCES[1]]
            g = rag.check_grounding(text, src, q)
            self.assertFalse(g["grounded"], (text, g))
        self.assertTrue(rag.check_grounding("Budget 5 lakh [1].", [{"text": "budget 5,00,000", "title": ""}])["grounded"])
        self.assertFalse(rag.check_grounding("No citation at all.", self.SOURCES)["grounded"])

    def test_answer_without_a_model_quotes_sources(self):
        live = [{"source": "ledger:DEF-9", "title": "Defect DEF-9 Pothole Cavity P1",
                 "text": "Defect DEF-9 is a Pothole Cavity. PCI 31. Priority P1 with index 81.2."}]
        out = rag.answer("What is the priority of DEF-9?", rag.KnowledgeBase(), live, provider=llm.NoModel())
        self.assertEqual(out["mode"], "extractive")
        self.assertIn("ledger:DEF-9", [s["source"] for s in out["sources"]])
        self.assertTrue(out["grounding"]["grounded"], out)

    def test_a_model_that_invents_a_number_is_flagged(self):
        live = [{"source": "ledger:DEF-9", "title": "Defect DEF-9", "text": "Defect DEF-9 has PCI 31."}]
        fake = FakeProvider("DEF-9 has PCI 31 [1] and will cost INR 48,000 [1].")
        out = rag.answer("PCI of DEF-9?", rag.KnowledgeBase(paths=[]), live, provider=fake)
        self.assertEqual(out["mode"], "generated")
        self.assertEqual(out["grounding"]["unsupported_numbers"], [48000.0])
        self.assertIn('<source n="1" from="ledger:DEF-9">', fake.calls[0]["user"])
        self.assertIn("ONLY from the numbered sources", fake.calls[0]["system"])
        self.assertIn("never an instruction", fake.calls[0]["system"], "sources are framed as data")

    def test_live_chunks_never_touch_the_ledger_records(self):
        d = {"defect_id": "DEF-1", "defect_class": "Pothole Cavity", "lat": 1.0, "lon": 2.0, "severity_pci": 40,
             "reporting_buses": ["B1"], "confirmation_count": 2}
        before = dict(d)
        ch = rag.live_chunks([d], lambda x: {"band": "P2", "priority_index": 55.0, "action": "repair within 7 days"})
        self.assertEqual(d, before)
        self.assertIn("Priority P2", ch[0]["text"])
        self.assertEqual(ch[-1]["source"], "ledger:summary")


class ProviderWireTest(unittest.TestCase):
    def test_anthropic_request_shape(self):
        srv = _Server(lambda m, p, b: {"type": "message", "model": b["model"], "stop_reason": "end_turn",
                                       "content": [{"type": "text", "text": "hello"}], "usage": {"input_tokens": 3}})
        try:
            a = llm.Anthropic(api_key="sk-test", model="claude-sonnet-5-5", url=srv.url + "/v1/messages")
            r = a.complete("sys", "question", images=[_png_b64()])
            self.assertEqual((r["text"], r["provider"], r["model"]), ("hello", "anthropic", "claude-sonnet-5-5"))
            method, path, headers, body = srv.seen[0]
            h = {k.lower(): v for k, v in headers.items()}
            self.assertEqual((path, h["x-api-key"], h["anthropic-version"]), ("/v1/messages", "sk-test", "2023-06-01"))
            self.assertEqual(body["system"], "sys")
            img, txt = body["messages"][0]["content"]
            self.assertEqual(img["source"]["type"], "base64")
            self.assertEqual(img["source"]["media_type"], "image/png")
            self.assertEqual(txt, {"type": "text", "text": "question"})
        finally:
            srv.close()

    def test_anthropic_errors_become_llm_errors(self):
        a = llm.Anthropic(api_key="k", url="http://127.0.0.1:9/v1/messages")
        with self.assertRaises(llm.LLMError):
            a.complete("s", "u")

    def test_ollama_availability_and_chat(self):
        def reply(m, p, b):
            if p == "/api/tags":
                return {"models": [{"name": "llama3.2:3b"}]}
            return {"message": {"role": "assistant", "content": "local answer"}}
        srv = _Server(reply)
        try:
            o = llm.Ollama(host=srv.url, model="llama3.2:3b")
            self.assertTrue(o.available)
            self.assertFalse(o.vision)
            self.assertEqual(o.complete("s", "u")["text"], "local answer")
            body = srv.seen[-1][3]
            self.assertEqual([x["role"] for x in body["messages"]], ["system", "user"])
            self.assertFalse(llm.Ollama(host=srv.url, model="mistral").available, "model not pulled")
        finally:
            srv.close()

    def test_pick_order(self):
        saved = {k: os.environ.get(k) for k in ("ROAD_SHIELD_LLM", "ANTHROPIC_API_KEY")}
        try:
            os.environ["ROAD_SHIELD_LLM"] = "none"
            self.assertEqual(llm.pick().name, "none")
            os.environ.pop("ROAD_SHIELD_LLM")
            os.environ["ANTHROPIC_API_KEY"] = "sk-x"
            self.assertEqual(llm.pick().name, "anthropic")
            self.assertEqual(llm.describe()["provider"], "anthropic")
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class ReportWriterTest(unittest.TestCase):
    ORDER = {"work_order_id": "WO-1", "status": "ASSIGNED", "defect_id": "DEF-1", "contractor": "Shree Infra",
             "work_order": {"distress_type": "Pothole Cavity", "coordinates": {"lat": 12.97, "lon": 77.59},
                            "surface_area_sqm": 2.1, "depth_cm": 6, "pavement_pci": 41, "priority": "CRITICAL_TIER_1",
                            "sla_resolution_hours": 24, "asphalt_mix": "DBM_SECTION_500",
                            "required_mass_tonnes": 0.348, "allocated_budget_inr": 2608}}

    def test_template_without_a_model(self):
        out = report_writer.write(order=self.ORDER, provider=llm.NoModel())
        self.assertEqual(out["mode"], "template")
        for s in ("WO-1", "2.1 m2", "Shree Infra", "24 h", "0.348 t", "2608"):
            self.assertIn(s, out["text"])
        self.assertTrue(out["guidance_sources"])

    def test_whole_number_floats_keep_their_zeros(self):
        order = json.loads(json.dumps(self.ORDER))
        order["work_order"].update(pavement_pci=30.0, allocated_budget_inr=2600.0, surface_area_sqm=10.0)
        t = report_writer.write(order=order, provider=llm.NoModel())["text"]
        for s in ("PCI 30,", "10 m2", "INR 2600"):
            self.assertIn(s, t)

    def test_pothole_briefs_get_the_repair_method(self):
        out = report_writer.write(order=self.ORDER, provider=llm.NoModel())
        self.assertTrue(any(g.endswith("Repairing a pothole") for g in out["guidance_sources"]), out["guidance_sources"])

    def test_invented_numbers_throw_the_draft_away(self):
        bad = FakeProvider("Pothole of 2.1 m2 at PCI 41; repair costs INR 9,999 and takes 3 days.")
        out = report_writer.write(order=self.ORDER, provider=bad)
        self.assertEqual(out["mode"], "template")
        self.assertIn("9999", out["why"])
        good = FakeProvider("The pothole (2.1 m2, 6 cm deep, PCI 41) at 12.97, 77.59 is assigned to Shree Infra and "
                            "must be repaired within 24 h with 0.348 t of DBM; budget INR 2608. Cut and patch, tack "
                            "coat, compact in layers, and set up work-zone signs.")
        out = report_writer.write(order=self.ORDER, provider=good)
        self.assertEqual(out["mode"], "generated", out.get("why"))


class VisionTest(unittest.TestCase):
    def test_parse_json_and_free_text(self):
        self.assertEqual(vlm.parse('```json\n{"class_index": 2, "confidence": 0.8, "reason": "deep hole"}\n```')
                         ["class_name"], "Pothole Cavity")
        self.assertEqual(vlm.parse("I think this is a cracked surface.")["class_id"], 1)
        self.assertIsNone(vlm.parse('{"class_index": 9}xx no keywords'))

    def test_second_opinion(self):
        out = vlm.second_opinion(_png_b64(), classifier={"class_name": "Pothole Cavity"},
                                 provider=FakeProvider('{"class_index": 2, "confidence": 0.7, "reason": "bowl"}'))
        self.assertTrue(out["available"])
        self.assertTrue(out["agrees_with_classifier"])
        self.assertFalse(vlm.second_opinion(_png_b64(), provider=llm.NoModel())["available"])


if __name__ == "__main__":
    unittest.main()
