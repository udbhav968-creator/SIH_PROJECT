import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.vercel_app import handler as _VercelHandler


class handler(_VercelHandler):
    pass
