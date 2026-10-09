"""
Generative AI for ROAD-SHIELD, kept on a short leash.

    llm.py            one interface over three providers, picked automatically:
                        Claude (ANTHROPIC_API_KEY set) -> Ollama on this machine -> no model (extractive)
    rag.py            retrieval-augmented answers: BM25 over the project's documents, a road-maintenance
                      knowledge base and the live ledger; answers cite their sources and every number in an
                      answer is checked against the sources it cites
    report_writer.py  engineer-ready text for a defect or a work order, written only from measured fields;
                      a draft that states a number not in those fields is thrown away for a template
    vlm.py            a vision-language "second opinion" on a photograph the classifier is unsure of; advisory,
                      never changes a measurement; scripts/measure_vlm.py scores it on held-out photographs

Nothing here changes a detection, a measurement, a priority or a work order.
"""
