"""Pure scoring helper shared by super_customer_score_v2.py and the API.

Deliberately has no data-loading or training side effects, unlike the
analysis scripts — safe to import in the FastAPI backend at startup without
retraining a model or needing data/cleaned_funnel_data.csv to exist (it
doesn't, on Railway; data/ is gitignored and never deployed).
"""

import numpy as np
import pandas as pd


def predict_super_customer_score(customer_df: pd.DataFrame, model) -> pd.DataFrame:
    """Score customers 0-100 on likelihood of becoming a super customer (referred=Yes).

    Returns a DataFrame (same index as `customer_df`) with a `Score` (0-100,
    rounded to 1 decimal) and a `Tier` (Low < 40, Medium 40-75, High > 75).
    """
    probability = model.predict_proba(customer_df)[:, 1]
    score = np.round(probability * 100, 1)
    tier = np.select(
        [score < 40, score <= 75],
        ["Low", "Medium"],
        default="High",
    )
    return pd.DataFrame({"Score": score, "Tier": tier}, index=customer_df.index)
