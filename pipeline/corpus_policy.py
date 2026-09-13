"""
Which photographs belong to the road-scene corpus, and which do not.

The problem
-----------
`datasets/03_crack500_fatigue` and `datasets/05_morth_civil_hard_negatives`
contain 622 files from a concrete surface-crack dataset: 227x227 close-ups of
cracked plaster and concrete walls. They are correctly labelled - a crack is a
crack - and they are the wrong domain.

ROAD-SHIELD audits a carriageway from a vehicle. Every rule it applies assumes
that: the road surface is in the lower part of the frame, a repair has an area
in square metres derived from camera geometry, and a proposal whose centroid
sits above the horizon fraction is discarded. A 227x227 texture patch has no
horizon, no perspective and no road, so those rules cannot apply to it. It is
neither detected nor detectable, and counting it as a missed road defect
measures the wrong thing.

Leaving it in the negatives is the more harmful half. A pixel classifier
trained to call cracked plaster "clean road" is being taught, with 622
examples, that a high-contrast linear feature on a grey surface is not a
defect - which is precisely the feature a road crack presents.

The decision
------------
Excluded from road-scene training and from road-scene measurement. Not
deleted, not hidden: the files stay on disk, the exclusion is listed with its
reason in every report that depends on it, and anyone can reproduce the
unfiltered number by setting ROAD_SHIELD_NO_CORPUS_FILTER=1.

Why by source rather than by size
---------------------------------
The obvious rule - drop anything smaller than 240 px - is wrong here. 325 of
489 photographs in the pothole folder are annotated crops below that size and
are genuine road imagery. Size does not separate the two; provenance does, and
the ingest already records provenance in the filename prefix it assigns.
"""

import os

# prefix -> why it is not a road scene. Keep the reason publishable.
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
