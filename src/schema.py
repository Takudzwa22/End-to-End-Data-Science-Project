"""Schema for Berlin rental listings.

The Immowelt Berlin listings Kaggle dataset may vary in column naming.
We infer the most likely columns from the CSV headers so the pipeline runs
with minimal manual edits.

If inference is wrong, hardcode the *_COL constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# ---- Defaults (used if inference fails) ----
TARGET_COL = "price"
AREA_COL = "area"
ROOMS_COL = "rooms"
DISTRICT_COL = "district"
YEAR_BUILT_COL = "year_built"
PROPERTY_TYPE_COL = "type"


@dataclass(frozen=True)
class ResolvedSchema:
    target: str
    area: Optional[str]
    rooms: Optional[str]
    district: Optional[str]
    year_built: Optional[str]
    property_type: Optional[str]


def _norm(s: str) -> str:
    s = s.strip().lower()
    for ch in [" ", "-", "_", "/", ".", ":", "(", ")", "[", "]"]:
        s = s.replace(ch, "")
    s = s.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    return s


def _pick(columns: Iterable[str], prefer: list[str], contains_any: list[str]) -> Optional[str]:
    cols = list(columns)
    norm_map = {_norm(c): c for c in cols}

    # 1) exact preferred names (after normalisation)
    for p in prefer:
        np = _norm(p)
        if np in norm_map:
            return norm_map[np]

    # 2) fallback: substring match scoring
    best: Optional[str] = None
    best_score = 0
    for c in cols:
        nc = _norm(c)
        score = 0
        for tok in contains_any:
            if _norm(tok) in nc:
                score += 1
        if score > best_score:
            best_score = score
            best = c
    return best if best_score > 0 else None


def resolve_from_columns(columns: Iterable[str]) -> ResolvedSchema:
    target = _pick(
        columns,
        prefer=[TARGET_COL, "kaltmiete", "cold_rent", "net_rent", "miete", "preis", "rent"],
        contains_any=["kalt", "miete", "rent", "preis", "price"],
    ) or TARGET_COL

    area = _pick(
        columns,
        prefer=[AREA_COL, "wohnflaeche", "living_area", "living_space", "flaeche", "sqm", "m2"],
        contains_any=["wohn", "flaeche", "area", "sqm", "m2"],
    )

    rooms = _pick(
        columns,
        prefer=[ROOMS_COL, "zimmer", "anzahl_zimmer"],
        contains_any=["zimmer", "room"],
    )

    district = _pick(
        columns,
        prefer=[DISTRICT_COL, "bezirk", "stadtteil", "ortsteil", "borough", "location"],
        contains_any=["bezirk", "district", "stadtteil", "ortsteil", "borough", "location"],
    )

    year_built = _pick(
        columns,
        prefer=[YEAR_BUILT_COL, "baujahr", "construction_year", "year_built"],
        contains_any=["baujahr", "construction", "built", "year"],
    )

    property_type = _pick(
        columns,
        prefer=[PROPERTY_TYPE_COL, "objektart", "property_type", "type"],
        contains_any=["objekt", "type", "property", "immobil"],
    )

    return ResolvedSchema(
        target=target,
        area=area,
        rooms=rooms,
        district=district,
        year_built=year_built,
        property_type=property_type,
    )


def resolve_inplace(columns: Iterable[str]) -> ResolvedSchema:
    """Resolve schema and update module-level constants."""
    global TARGET_COL, AREA_COL, ROOMS_COL, DISTRICT_COL, YEAR_BUILT_COL, PROPERTY_TYPE_COL

    s = resolve_from_columns(columns)
    TARGET_COL = s.target
    if s.area:
        AREA_COL = s.area
    if s.rooms:
        ROOMS_COL = s.rooms
    if s.district:
        DISTRICT_COL = s.district
    if s.year_built:
        YEAR_BUILT_COL = s.year_built
    if s.property_type:
        PROPERTY_TYPE_COL = s.property_type
    return s
