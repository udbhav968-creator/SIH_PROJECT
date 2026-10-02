import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.vercel_app import handler  # noqa: F401  (Vercel looks for `handler`)
