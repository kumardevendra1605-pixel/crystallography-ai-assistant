from collections import OrderedDict

# How many keywords two sources need to share before we group them together
MIN_OVERLAP = 2

# Words that are too generic to use as group labels
_GENERIC_WORDS = {
    "crystal", "crystals", "data", "sample", "structure", "using",
    "used", "also", "make", "need", "good", "best", "different",
    "possible", "problem", "question", "answer", "know", "think",
    "would", "could", "should", "have", "does", "will", "your",
    "about", "there", "their", "they", "been", "were",
}

# Nice human-readable names for common topic combinations
TOPIC_LABELS = {
    frozenset({"phase", "problem"}):         "Phase Problem",
    frozenset({"phase", "information"}):     "Phase Information",
    frozenset({"twin", "twinning"}):         "Twinning",
    frozenset({"twin", "detection"}):        "Twinning — Detection",
    frozenset({"twin", "refinement"}):       "Twinning — Refinement",
    frozenset({"disorder", "refinement"}):   "Disorder & Refinement",
    frozenset({"absorption", "correction"}): "Absorption Correction",
    frozenset({"diffraction", "weak"}):      "Weak Diffraction",
    frozenset({"resolution", "data"}):       "Data Resolution",
    frozenset({"signal", "noise"}):          "Signal-to-Noise Ratio",
    frozenset({"omega", "phi"}):             "Goniometer Axes",
    frozenset({"radiation", "damage"}):      "Radiation Damage",
    frozenset({"space", "group"}):           "Space Group",
    frozenset({"unit", "cell"}):             "Unit Cell",
    frozenset({"hydrogen", "bond"}):         "Hydrogen Bonding",
    frozenset({"electron", "density"}):      "Electron Density",
    frozenset({"refinement", "restraint"}):  "Refinement & Restraints",
    frozenset({"completeness", "data"}):     "Data Completeness",
    frozenset({"wavelength", "radiation"}):  "X-ray Wavelength",
    frozenset({"crystal", "growth"}):        "Crystal Growth",
}


def group_by_subtopic(sources, query=""):
    """Cluster a list of source entries into related sub-topic groups.

    Uses keyword overlap as the primary signal — two sources get grouped
    together if they share at least MIN_OVERLAP meaningful keywords.

    With 1-2 sources we just return a single "General" group since there's
    nothing meaningful to cluster. Groups are ordered by their highest
    confidence score so the most relevant topic comes first.
    """
    if not sources:
        return OrderedDict()

    if len(sources) <= 2:
        return OrderedDict([("General", list(sources))])

    # Build a keyword set for each source, filtering out generic words
    kw_sets = [
        {w for w in src.get("keywords", []) if w not in _GENERIC_WORDS}
        for src in sources
    ]

    # Greedy clustering — assign each source to the first cluster it fits
    assigned = [False] * len(sources)
    clusters = []

    for i in range(len(sources)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(sources)):
            if assigned[j]:
                continue
            if len(kw_sets[i] & kw_sets[j]) >= MIN_OVERLAP:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Sort by the highest confidence in each cluster
    clusters.sort(key=lambda c: max(sources[i]["confidence"] for i in c), reverse=True)

    result = OrderedDict()
    used_labels = set()

    for cluster_indices in clusters:
        cluster_sources = [sources[i] for i in cluster_indices]
        label = _pick_label(cluster_sources, kw_sets, cluster_indices, used_labels)
        used_labels.add(label)
        result[label] = cluster_sources

    return result


def _pick_label(cluster_sources, kw_sets, indices, used_labels):
    """Come up with a readable name for a cluster of sources.

    First checks if any pair of top keywords matches our predefined label map.
    Falls back to title-casing the top keywords if nothing matches.
    """
    # Count keyword frequency across the whole cluster
    freq = {}
    for i in indices:
        for w in kw_sets[i]:
            freq[w] = freq.get(w, 0) + 1

    top_words = sorted(freq, key=lambda w: -freq[w])

    # Try to find a nice label from our predefined map
    for a in top_words[:5]:
        for b in top_words[:5]:
            if a == b:
                continue
            candidate = TOPIC_LABELS.get(frozenset({a, b}))
            if candidate and candidate not in used_labels:
                return candidate

    # Fall back to title-casing the top keywords
    label_words = [w.title() for w in top_words[:3] if w not in _GENERIC_WORDS]
    if label_words:
        label = " — ".join(label_words[:2]) if len(label_words) >= 2 else label_words[0]
    else:
        label = "General"

    # Make sure we don't have duplicate labels
    base = label
    n = 2
    while label in used_labels:
        label = f"{base} ({n})"
        n += 1

    return label
