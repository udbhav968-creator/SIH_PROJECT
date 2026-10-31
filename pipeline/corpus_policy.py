"""Data retention and inspection corpus governance policy."""

import os

NON_ROAD_SOURCES = {
    "kag_surface-crack": (
        "227x227 close-ups of cracked concrete and plaster surfaces (SDNET-style "
        "texture patches). No road surface, no horizon and no camera geometry, so "
        "neither the region-of-interest rule nor the area model applies."),
}

DISABLE_ENV = "ROAD_SHIELD_NO_CORPUS_FILTER"


def filtering_enabled():
    return os.environ.get(DISABLE_ENV) != "1"


def non_road_source(path):
    """The prefix that excludes this file, or None."""
    name = os.path.basename(path)
    for prefix in NON_ROAD_SOURCES:
        if name.startswith(prefix):
            return prefix
    return None


def is_road_scene(path):
    if not filtering_enabled():
        return True
    return non_road_source(path) is None


def filter_paths(paths):
    """(kept, {prefix: how many dropped})."""
    if not filtering_enabled():
        return list(paths), {}
    kept, dropped = [], {}
    for p in paths:
        pre = non_road_source(p)
        if pre is None:
            kept.append(p)
        else:
            dropped[pre] = dropped.get(pre, 0) + 1
    return kept, dropped


def policy_record(dropped=None):
    """What to write into a report so the exclusion travels with the number."""
    return {
        "enabled": filtering_enabled(),
        "excluded_sources": {k: {"reason": v, "files_dropped": (dropped or {}).get(k, 0)}
                             for k, v in NON_ROAD_SOURCES.items()},
        "reproduce_unfiltered_with": f"{DISABLE_ENV}=1",
        "not_deleted": "The files remain on disk; only this corpus excludes them.",
    }
