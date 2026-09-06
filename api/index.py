import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

from api.server import RoadShieldAPIHandler

class handler(RoadShieldAPIHandler):
    pass
