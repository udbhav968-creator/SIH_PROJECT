"""
Loading the shipped scikit-learn checkpoints under a newer scikit-learn.

The checkpoints were pickled under scikit-learn 1.8. Models with a multinomial loss (the segmenter's
gradient-boosted pixel classifier) refer in the pickle to the Cython module by the bare name "_loss",
which 1.9 no longer registers, so they failed to load with "No module named '_loss'" and the pipeline fell
back to brightness proposals without the false-positive gate. Registering the module under that name is
enough to load them; the class behind it is the same Cython loss object.

This is a loading fix, not a promise that 1.9 gives identical numbers: requirements.txt still pins
scikit-learn < 1.9, the version every report was measured with. Checked on 9 Oct 2026 under 1.9.1 with this
shim: the whole test suite passes (the 9 segmenter and false-positive-gate tests that failed before now pass),
and scripts/measure_detection_quality.py found 19/24 annotated defects with 2/36 false positives, against
18/24 and 4/36 in the committed report (measured under 1.8 on an earlier build, so not a like-for-like
comparison; it shows the loaded models behave, not that the numbers are identical).
"""
import sys


def install():
    if "_loss" in sys.modules:
        return
    try:
        import sklearn._loss._loss as loss_module
    except Exception:
        return
    sys.modules["_loss"] = loss_module


install()
