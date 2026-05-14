"""
UFI Project — Dataset Generator
Generates 2,520 road-hour records: 105 roads × 10 neighbourhoods × 24 hours
matching the specification in the project document.
"""

import numpy as np
import pandas as pd
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Neighbourhoods and road segments ─────────────────────────────────────────
NEIGHBOURHOODS = [
    "Nungambakkam", "Velachery", "T. Nagar", "Anna Nagar",
    "Adyar", "Mylapore", "Tambaram", "Chromepet", "Perambur", "Guindy"
]

ROADS_PER_NEIGHBOURHOOD = {n: 10 for n in NEIGHBOURHOODS}   # 10 × 10 = 100 ... +5 = 105
ROADS_PER_NEIGHBOURHOOD["Nungambakkam"] += 5                 # busiest area, extra roads

# Congestion profile: base speed ratio  speed / speed_limit
CONGESTION_PROFILES = {
    "Nungambakkam": 0.45,
    "Velachery":    0.48,
    "T. Nagar":     0.52,
    "Anna Nagar":   0.60,
    "Adyar":        0.62,
    "Mylapore":     0.58,
    "Tambaram":     0.65,
    "Chromepet":    0.63,
    "Perambur":     0.68,
    "Guindy":       0.55,
}

# Rush-hour multipliers (hour → speed penalty)
def rush_factor(hour: int) -> float:
    """Return a speed-reduction multiplier for a given hour (0–23)."""
    if 7 <= hour <= 10:       # morning peak
        return 0.55
    elif 17 <= hour <= 20:    # evening peak
        return 0.60
    elif 12 <= hour <= 14:    # lunch
        return 0.80
    elif 0 <= hour <= 5:      # night / free flow
        return 1.10
    else:
        return 0.90

# Incident text pool (for NLP pipeline)
HIGH_SEV  = ["fatal accident blocked road", "major collision road closed", "truck overturned blocking all lanes"]
MED_SEV   = ["accident reported minor injuries", "construction work causing delays", "vehicle breakdown lane blocked"]
LOW_SEV   = ["minor fender bender cleared", "slow traffic near junction", "slight congestion due to signal"]
NO_INC    = ["no incidents reported", "traffic moving normally", "clear roads"]

INCIDENT_POOL = (
    [(t, 3) for t in HIGH_SEV] +
    [(t, 2) for t in MED_SEV]  +
    [(t, 1) for t in LOW_SEV]  +
    [(t, 0) for t in NO_INC]
)


def sample_incident(p_incident: float):
    """Sample an incident text and its raw weight."""
    if random.random() > p_incident:
        text, weight = random.choice([(t, 0) for t in NO_INC])
    else:
        weights = [3, 3, 3, 2, 2, 2, 1, 1, 1]  # skewed towards lower severity
        pool    = HIGH_SEV + MED_SEV + LOW_SEV
        text    = random.choice(pool)
        weight  = 3 if text in HIGH_SEV else (2 if text in MED_SEV else 1)
    return text, weight


def build_dataset() -> pd.DataFrame:
    records = []
    road_id = 1

    for neighbourhood, n_roads in ROADS_PER_NEIGHBOURHOOD.items():
        base_speed_ratio = CONGESTION_PROFILES[neighbourhood]

        for road_num in range(1, n_roads + 1):
            speed_limit = random.choice([40, 50, 60, 80])  # km/h
            capacity    = random.randint(800, 2000)         # veh/hour

            for hour in range(24):
                rf            = rush_factor(hour)
                speed_ratio   = min(base_speed_ratio * rf, 1.0)
                noise         = np.random.normal(0, 0.05)
                speed_ratio   = np.clip(speed_ratio + noise, 0.15, 1.0)
                avg_speed     = round(speed_limit * speed_ratio, 1)

                # Volume rises during peak, drops at night
                v_ratio       = 1.0 / rf                    # inverse: slower → more cars
                volume        = int(capacity * v_ratio * np.random.uniform(0.6, 0.95))
                volume        = min(volume, capacity)

                p_incident    = 0.5 * (1 - speed_ratio)     # more congestion → more incidents
                incident_text, inc_weight = sample_incident(p_incident)
                incident_count = 1 if inc_weight > 0 else 0

                records.append({
                    "road_id":         f"R{road_id:03d}",
                    "road_name":       f"{neighbourhood}_Rd_{road_num}",
                    "neighbourhood":   neighbourhood,
                    "hour":            hour,
                    "speed_limit":     speed_limit,
                    "avg_speed":       avg_speed,
                    "volume":          volume,
                    "capacity":        capacity,
                    "incident_count":  incident_count,
                    "incident_text":   incident_text,
                    "incident_weight": inc_weight,
                })

            road_id += 1

    df = pd.DataFrame(records)
    df.to_csv("data/ufi_raw.csv", index=False)
    print(f"Dataset saved: {len(df)} records  |  {df['road_id'].nunique()} roads  |  {df['neighbourhood'].nunique()} neighbourhoods")
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(df.head(10).to_string())
