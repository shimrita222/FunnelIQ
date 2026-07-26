"""Repeatable loader: data/funnel_marketing_data.csv -> Supabase `customers` table.

Safe to re-run: it clears the table before inserting, so it always ends up
with exactly one fresh copy of the CSV.
"""

import math

import pandas as pd

from backend.supabase_client import get_supabase_admin_client

CSV_PATH = "data/funnel_marketing_data.csv"
BATCH_SIZE = 500


def load_records() -> list[dict]:
    df = pd.read_csv(CSV_PATH)
    df["purchased"] = df["purchased"].astype(bool)
    df["upsell"] = df["upsell"].astype(bool)
    df["referred"] = df["referred"].map({"Yes": True, "No": False})
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records


def main() -> None:
    records = load_records()
    supabase = get_supabase_admin_client()

    supabase.table("customers").delete().gt("id", 0).execute()

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        supabase.table("customers").insert(batch).execute()
        print(f"Inserted rows {i} to {i + len(batch)}")

    print(f"Done: {len(records)} rows loaded into customers")


if __name__ == "__main__":
    main()
