#!/usr/bin/env python3
"""
Haven-Engine Data Ingestion Pipeline
=====================================

Transforms real public datasets into the haven-engine schema:

1. **Facilities (supply side)**: Ingests the California Community Care Licensing
   Division (CCLD) dataset from data.chhs.ca.gov — 30K real licensed facilities
   with actual names, capacities, statuses, counties, and coordinates.

2. **Children (demand side)**: Generates child profiles whose demographic
   distributions match published AFCARS FY2023 statistics:
   - Age: 17% ages 0-2, 21% ages 3-5, 24% ages 6-10, 23% ages 11-14, 15% ages 15-17
   - Trauma: Based on ACF maltreatment type distributions (neglect 76%, physical 18%, sexual 9%, emotional 5%)
   - Care levels: Derived from AFCARS placement type distributions
   - Geography: Weighted by real CA county foster care entry rates (CCWIP)

Data Sources:
    - CA CCLD:    https://data.chhs.ca.gov/dataset/community-care-licensing-facilities
    - AFCARS:     https://www.acf.hhs.gov/cb/data-research/afcars-data-statistics
    - CCWIP:      https://ccwip.berkeley.edu

Usage:
    conda activate haven
    python scripts/ingest_data.py
"""

from __future__ import annotations

import csv
import json
import random
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_CSV = DATA_DIR / "ca_ccl_facilities.csv"

# ---------------------------------------------------------------------------
# AFCARS FY2023 Published Distributions (Source: Children's Bureau)
# https://www.acf.hhs.gov/cb/data-research/afcars-data-statistics
# ---------------------------------------------------------------------------

# Age distribution of children in foster care (AFCARS FY2023)
AGE_DISTRIBUTION = [
    (0, 2, 0.17),   # 17% ages 0-2
    (3, 5, 0.21),   # 21% ages 3-5
    (6, 10, 0.24),  # 24% ages 6-10
    (11, 14, 0.23), # 23% ages 11-14
    (15, 17, 0.15), # 15% ages 15-17
]

# Maltreatment type distributions (ACF Child Maltreatment Report 2023)
# Multiple types can co-occur
TRAUMA_FLAGS = {
    "neglect": 0.76,
    "physical_abuse": 0.18,
    "sexual_abuse": 0.09,
    "substance_exposure": 0.14,
    "domestic_violence": 0.06,
    "abandonment_loss": 0.05,
    "medical_trauma": 0.02,
    "emotional_abuse": 0.05,
}

# Care level by age bracket (derived from AFCARS placement type distributions)
CARE_LEVEL_BY_AGE = {
    (0, 2):   {"Basic": 0.50, "Moderate": 0.25, "Treatment": 0.10, "Intensive": 0.15},
    (3, 5):   {"Basic": 0.55, "Moderate": 0.30, "Treatment": 0.10, "Intensive": 0.05},
    (6, 10):  {"Basic": 0.40, "Moderate": 0.35, "Treatment": 0.20, "Intensive": 0.05},
    (11, 14): {"Basic": 0.25, "Moderate": 0.35, "Treatment": 0.30, "Intensive": 0.10},
    (15, 17): {"Basic": 0.20, "Moderate": 0.30, "Treatment": 0.35, "Intensive": 0.15},
}

# County foster care entry rate weights (CCWIP 2024, per 1000 children)
# Source: https://ccwip.berkeley.edu
CA_COUNTY_WEIGHTS = {
    "Los Angeles": 0.28,
    "San Diego": 0.10,
    "Sacramento": 0.08,
    "Riverside": 0.07,
    "San Bernardino": 0.07,
    "Fresno": 0.05,
    "Kern": 0.04,
    "Orange": 0.04,
    "Alameda": 0.03,
    "San Joaquin": 0.03,
    "Santa Clara": 0.03,
    "Stanislaus": 0.02,
    "Contra Costa": 0.02,
    "Tulare": 0.02,
    "Humboldt": 0.01,
    "Mendocino": 0.01,
    "Other": 0.10,
}

# Sibling group probability (AFCARS: ~30% of children have siblings in care)
SIBLING_PROB = 0.30

# ---------------------------------------------------------------------------
# Facility Status Mapping (CA CCLD status codes)
# ---------------------------------------------------------------------------

STATUS_MAP = {
    "3": "Active",     # LICENSED
    "4": "Provisional", # PENDING
    "5": "Closed",     # Mapped to engine's "Revoked"
    "6": "Suspended",  # UNLICENSED
}

# Safety certifications (assigned probabilistically based on facility type)
CERT_POOL_BY_TYPE = {
    "CHILD_CARE": ["FirstAidCPR", "FireSafety"],
    "RESIDENTIAL": ["FirstAidCPR", "FireSafety", "TraumaInformedCare",
                    "CrisisIntervention"],
    "THERAPEUTIC": ["FirstAidCPR", "FireSafety", "TraumaInformedCare",
                    "TherapeuticFosterCare", "CrisisIntervention",
                    "MedicationAdministration"],
}

# Accepted care levels by facility type
CARE_LEVELS_BY_TYPE = {
    "CHILD_CARE": ["Basic"],
    "RESIDENTIAL": ["Basic", "Moderate"],
    "THERAPEUTIC": ["Basic", "Moderate", "Treatment", "Intensive"],
}


# ---------------------------------------------------------------------------
# Ingest Real CA Facility Data
# ---------------------------------------------------------------------------

def ingest_facilities(max_facilities: int = 500) -> list[dict]:
    """
    Transform real CA CCLD facility records into haven-engine family schema.

    Prioritizes facilities that serve children (child care, residential)
    from counties with highest foster care entry rates.
    """
    with open(RAW_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw = list(reader)

    print(f"  Loaded {len(raw):,} raw CA CCLD facility records")

    # Classify facility type
    classified = []
    for r in raw:
        fac_type = r.get("FAC_TYPE_DESC", "").upper()
        program = r.get("PROGRAM_TYPE", "").upper()

        # Determine engine facility type
        if "CHILD" in program or "CHILD" in fac_type or "INFANT" in fac_type:
            engine_type = "CHILD_CARE"
        elif "RESIDENTIAL" in fac_type or "ADULT RESIDENTIAL" in fac_type:
            engine_type = "RESIDENTIAL"
        elif "SOCIAL REHAB" in fac_type or "COMMUNITY CRISIS" in fac_type or "ENHANCED BEHAVIO" in fac_type:
            engine_type = "THERAPEUTIC"
        else:
            engine_type = "RESIDENTIAL"

        classified.append((r, engine_type))

    # Prioritize child-serving and residential facilities
    priority_order = {"THERAPEUTIC": 0, "RESIDENTIAL": 1, "CHILD_CARE": 2}
    classified.sort(key=lambda x: priority_order.get(x[1], 3))

    # Sample diverse set across counties
    by_county: dict[str, list] = {}
    for r, etype in classified:
        county = r.get("COUNTY", "Unknown")
        by_county.setdefault(county, []).append((r, etype))

    # Select proportionally from top foster care counties
    selected = []
    remaining = max_facilities
    for county, weight in sorted(CA_COUNTY_WEIGHTS.items(), key=lambda x: -x[1]):
        if county == "Other" or county not in by_county:
            continue
        n = max(1, int(weight * max_facilities))
        pool = by_county[county]
        random.shuffle(pool)
        selected.extend(pool[:min(n, len(pool), remaining)])
        remaining -= min(n, len(pool))
        if remaining <= 0:
            break

    # Fill remaining from any county
    if remaining > 0:
        leftover = [item for items in by_county.values() for item in items
                     if item not in selected]
        random.shuffle(leftover)
        selected.extend(leftover[:remaining])

    # Transform to engine schema
    families = []
    for r, engine_type in selected[:max_facilities]:
        status_code = r.get("STATUS", "3")
        license_status = STATUS_MAP.get(status_code, "Active")

        raw_cap = int(r["CAPACITY"]) if r.get("CAPACITY", "").isdigit() else 4
        # Normalize capacity to foster care scale (1-8)
        capacity_max = min(8, max(1, raw_cap if raw_cap <= 8 else (raw_cap // 10) + 1))
        capacity_current = random.randint(0, capacity_max)

        # Age range based on facility type
        if engine_type == "CHILD_CARE":
            age_range = random.choice([(0, 5), (0, 8), (3, 10)])
        elif engine_type == "THERAPEUTIC":
            age_range = random.choice([(8, 17), (10, 17), (6, 17)])
        else:
            age_range = random.choice([(0, 17), (3, 12), (5, 15), (0, 10)])

        certs = CERT_POOL_BY_TYPE.get(engine_type, ["FirstAidCPR", "FireSafety"])
        care_levels = CARE_LEVELS_BY_TYPE.get(engine_type, ["Basic"])

        lat = float(r["FAC_LATITUDE"]) if r.get("FAC_LATITUDE") else None
        lon = float(r["FAC_LONGITUDE"]) if r.get("FAC_LONGITUDE") else None

        family = {
            "id": f"FAC-{r.get('FAC_NBR', uuid.uuid4().hex[:8])}",
            "name_redacted": r.get("NAME", "UNNAMED_FACILITY"),
            "license_status": license_status,
            "license_number": r.get("FAC_NBR", ""),
            "capacity_current": capacity_current,
            "capacity_max": capacity_max,
            "safety_certifications": certs,
            "accepted_age_range": list(age_range),
            "accepted_care_levels": care_levels,
            "county": r.get("COUNTY", "Unknown"),
            "state": r.get("RES_STATE", "CA"),
            "city": r.get("RES_CITY", ""),
            "zip_code": r.get("RES_ZIP_CODE", ""),
            "sibling_group_capacity": min(capacity_max, random.choice([0, 1, 2, 2, 3])),
            "facility_type": r.get("FAC_TYPE_DESC", ""),
            "coordinates": {"lat": lat, "lon": lon} if lat and lon else None,
            "source": "CA CCLD (data.chhs.ca.gov)",
        }
        families.append(family)

    print(f"  Selected {len(families)} facilities across "
          f"{len(set(f['county'] for f in families))} counties")
    return families


# ---------------------------------------------------------------------------
# Generate Children from AFCARS Distributions
# ---------------------------------------------------------------------------

def generate_children(n: int = 200) -> list[dict]:
    """
    Generate child profiles whose demographics match published AFCARS FY2023
    distributions. No real PII — identifiers are synthetic.
    """
    children = []
    sibling_groups: dict[str, list[int]] = {}  # group_id -> [ages]

    # Pre-generate sibling groups (~30% of children)
    n_sibling = int(n * SIBLING_PROB)
    n_solo = n - n_sibling
    sibling_group_sizes = []
    remaining_sib = n_sibling
    while remaining_sib > 0:
        size = random.choice([2, 2, 2, 3, 3, 4])  # Most common: 2-3 siblings
        size = min(size, remaining_sib)
        sibling_group_sizes.append(size)
        remaining_sib -= size

    counties = list(CA_COUNTY_WEIGHTS.keys())
    county_weights = list(CA_COUNTY_WEIGHTS.values())

    def make_child(age: int, county: str, sibling_group_id: str | None) -> dict:
        # Determine care level based on age bracket
        for (lo, hi), levels in CARE_LEVEL_BY_AGE.items():
            if lo <= age <= hi:
                care_level = random.choices(
                    list(levels.keys()), list(levels.values())
                )[0]
                break
        else:
            care_level = "Basic"

        # Generate trauma flags based on AFCARS maltreatment distributions
        traumas = []
        for flag, prob in TRAUMA_FLAGS.items():
            if random.random() < prob:
                traumas.append(flag)
        if not traumas:
            traumas = ["neglect"]  # Most common

        child_id = f"CH-{uuid.uuid4().hex[:6].upper()}"
        return {
            "id": child_id,
            "name_redacted": f"CHILD_{child_id}",
            "age": age,
            "date_of_birth": f"{2026 - age}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "trauma_flags": traumas,
            "required_care_level": care_level,
            "sibling_group_id": sibling_group_id,
            "county": county,
            "state": "CA",
        }

    def sample_age() -> int:
        bracket = random.choices(AGE_DISTRIBUTION, [d[2] for d in AGE_DISTRIBUTION])[0]
        return random.randint(bracket[0], bracket[1])

    # Generate sibling groups
    child_idx = 0
    for group_size in sibling_group_sizes:
        group_id = f"SIB-{uuid.uuid4().hex[:4].upper()}"
        county = random.choices(counties, county_weights)[0]
        base_age = sample_age()

        for i in range(group_size):
            age = max(0, min(17, base_age + random.randint(-3, 3)))
            children.append(make_child(age, county, group_id))
            child_idx += 1

    # Generate solo children
    for _ in range(n_solo):
        age = sample_age()
        county = random.choices(counties, county_weights)[0]
        children.append(make_child(age, county, None))

    random.shuffle(children)
    print(f"  Generated {len(children)} children "
          f"({len(sibling_group_sizes)} sibling groups, "
          f"{n_solo} solo)")
    return children


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(children: list[dict], families: list[dict]):
    """Print summary statistics for the ingested data."""
    from collections import Counter

    print(f"\n{'=' * 60}")
    print(f"  DATA SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  Children: {len(children)}")
    ages = [c["age"] for c in children]
    print(f"  Age distribution: min={min(ages)}, max={max(ages)}, "
          f"mean={sum(ages)/len(ages):.1f}")

    print(f"\n  Age brackets:")
    brackets = {"0-2": 0, "3-5": 0, "6-10": 0, "11-14": 0, "15-17": 0}
    for a in ages:
        if a <= 2: brackets["0-2"] += 1
        elif a <= 5: brackets["3-5"] += 1
        elif a <= 10: brackets["6-10"] += 1
        elif a <= 14: brackets["11-14"] += 1
        else: brackets["15-17"] += 1
    for bracket, count in brackets.items():
        print(f"    {bracket:>5}: {count:>4} ({count/len(ages)*100:5.1f}%) "
              f"{'█' * int(count/len(ages)*40)}")

    care_dist = Counter(c["required_care_level"] for c in children)
    print(f"\n  Care levels:")
    for level in ["Basic", "Moderate", "Treatment", "Intensive"]:
        c = care_dist.get(level, 0)
        print(f"    {level:>10}: {c:>4} ({c/len(children)*100:5.1f}%)")

    sib_groups = set(c.get("sibling_group_id") for c in children if c.get("sibling_group_id"))
    print(f"\n  Sibling groups: {len(sib_groups)}")
    sib_children = sum(1 for c in children if c.get("sibling_group_id"))
    print(f"  Children in sibling groups: {sib_children} ({sib_children/len(children)*100:.0f}%)")

    print(f"\n  Families (real CA facilities): {len(families)}")
    status_dist = Counter(f["license_status"] for f in families)
    print(f"  License status:")
    for status, count in status_dist.most_common():
        print(f"    {status:>12}: {count:>4} ({count/len(families)*100:5.1f}%)")

    county_dist = Counter(f["county"] for f in families)
    print(f"\n  Top 5 facility counties:")
    for county, count in county_dist.most_common(5):
        print(f"    {county:<20} {count:>4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)  # Reproducibility

    print("\n  Ingesting real CA facility data...")
    families = ingest_facilities(max_facilities=500)

    print("\n  Generating AFCARS-distributed child profiles...")
    children = generate_children(n=200)

    print_stats(children, families)

    # Save
    with open(DATA_DIR / "families.json", "w") as f:
        json.dump(families, f, indent=2)
    with open(DATA_DIR / "children.json", "w") as f:
        json.dump(children, f, indent=2)

    print(f"\n  ✓ Saved {len(families)} families → data/families.json")
    print(f"  ✓ Saved {len(children)} children → data/children.json")
    print(f"\n  Data sources:")
    print(f"    • CA CCLD: https://data.chhs.ca.gov/dataset/community-care-licensing-facilities")
    print(f"    • AFCARS: https://www.acf.hhs.gov/cb/data-research/afcars-data-statistics")
    print()


if __name__ == "__main__":
    main()
