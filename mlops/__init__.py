"""
MLOps for ROAD-SHIELD: the life of a model after its training script finishes.

    tracking.py         every training run: parameters, metrics, the files it wrote, the commit
    registry.py         versioned models with content hashes, stages and promotion gates;
                        deploys a version into checkpoints/ and rolls it back
    specs.py            which files make up each model, how it is scored, how it is retrained
    monitor.py          what the served models see in production: input drift, prediction drift,
                        out-of-distribution rate, latency, and a shadow candidate's agreement
    active_learning.py  the photographs the models are least sure about, queued for labelling
    retrain.py          validate data -> train -> evaluate -> register -> gate -> promote

Command line: python -m mlops --help
"""
