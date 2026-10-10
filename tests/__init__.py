"""Test package. Public-API lookups (rain, road: services/road_context.py) stay off for the whole suite, so
no test waits on, or depends on, the network. Tests that cover them inject a fake fetcher."""
import os

os.environ["ROAD_SHIELD_CONTEXT_APIS"] = "0"
