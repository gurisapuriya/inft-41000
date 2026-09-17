"""
fraudguard_datagen.py
----------------------
Shared synthetic-data generator for the FraudGuard case study (INFT 41000,
Weeks 1-14). Produces a PaySim-style mobile-money transaction dataset with
injectable class imbalance, missing values, and (from Week 10 onward)
volume spikes and concept drift.

This file is intentionally dependency-light (numpy + pandas only) so it
runs anywhere the course's requirements.txt is installed.

Reuse: copy this file into each new week's folder, or add
`Week01_Introduction_to_Production_AI/` to your PYTHONPATH so later weeks
can `from fraudguard_datagen import generate_transactions`.

Schema (mirrors the well-known PaySim synthetic fraud dataset):
    step            int    simulated time step (1 step = 1 hour)
    type            str    CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount          float  transaction amount
    nameOrig        str    origin account id
    oldbalanceOrg   float  origin balance before transaction
    newbalanceOrig  float  origin balance after transaction
    nameDest        str    destination account id
    oldbalanceDest  float  destination balance before transaction
    newbalanceDest  float  destination balance after transaction
    isFraud         int    1 if the transaction is fraudulent, else 0
    isFlaggedFraud  int    1 if a simple rule-based system would have
                           flagged it (large TRANSFER draining an account)
"""
import numpy as np
import pandas as pd

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
# Real-world-ish mix: payments and cash-outs dominate; transfers are rarer.
TYPE_WEIGHTS = [0.22, 0.35, 0.06, 0.30, 0.07]
# Fraud in PaySim-style data only ever hides inside these two types.
FRAUD_ELIGIBLE_TYPES = {"TRANSFER", "CASH_OUT"}


def _account_id(prefix, n):
    return np.array([f"{prefix}{i:09d}" for i in n])


def generate_transactions(
    n_rows=200_000,
    fraud_rate=0.006,
    missing_rate=0.001,
    seed=42,
    n_steps=30 * 24,
    drift=False,
    volume_multiplier=1.0,
):
    """
    Generate a synthetic FraudGuard transactions DataFrame.

    Parameters
    ----------
    n_rows : int
        Number of transactions to generate (before any volume_multiplier).
    fraud_rate : float
        Target proportion of rows that are fraudulent. NOTE: real-world
        PaySim fraud rate is ~0.13% (extremely rare). We deliberately use
        a higher rate (default 0.6%) in early weeks so a small teaching
        dataset still contains enough fraud examples to learn from. We
        revisit *realistic* extreme imbalance in Week 9 (fairness) and
        Week 11 (performance under rare-event constraints).
    missing_rate : float
        Proportion of balance-column values deliberately set to NaN, to
        give you real (not hypothetical) missing-data errors to fix.
    seed : int
        Random seed. IMPORTANT: this seeds BOTH numpy and Python's
        built-in random-derived choices used here, so the dataset is
        reproducible run to run. Forgetting to seed one or the other is a
        classic reproducibility bug -- see "Expected Errors & Fixes".
    n_steps : int
        Number of simulated hourly time steps transactions are spread
        across.
    drift : bool
        If True, shift the fraud pattern partway through the timeline
        (higher fraud rate + larger fraud amounts in the back half) to
        simulate concept drift. Used from Week 13 onward -- left False by
        default in Week 1.
    volume_multiplier : float
        Scales n_rows up to simulate traffic spikes (used in Week 10).

    Returns
    -------
    pandas.DataFrame
    """
    rng = np.random.default_rng(seed)
    n = int(n_rows * volume_multiplier)

    steps = rng.integers(1, n_steps + 1, size=n)
    types = rng.choice(TRANSACTION_TYPES, size=n, p=TYPE_WEIGHTS)

    # Base (legitimate-looking) amounts: log-normal, realistic long tail.
    amounts = rng.lognormal(mean=6.5, sigma=1.3, size=n)

    orig_ids = _account_id("C", rng.integers(1, n // 3 + 2, size=n))
    dest_ids = _account_id("C", rng.integers(1, n // 3 + 2, size=n))

    old_bal_orig = rng.lognormal(mean=8.0, sigma=1.5, size=n)
    old_bal_dest = rng.lognormal(mean=7.5, sigma=1.6, size=n)

    new_bal_orig = np.clip(old_bal_orig - amounts, 0, None)
    new_bal_dest = old_bal_dest + amounts

    is_fraud = np.zeros(n, dtype=int)
    fraud_eligible_mask = np.isin(types, list(FRAUD_ELIGIBLE_TYPES))
    eligible_idx = np.where(fraud_eligible_mask)[0]
    # fraud_rate is the TARGET OVERALL proportion of all rows (matching the
    # docstring), but fraud only ever hides inside eligible types, so the
    # per-eligible-row probability must be scaled up accordingly.
    eligible_fraction = max(len(eligible_idx) / n, 1e-9)
    base_rate = min(fraud_rate / eligible_fraction, 1.0)

    if drift:
        # Back half of the timeline: fraud becomes more common AND bigger.
        back_half_boost = np.where(steps[eligible_idx] > n_steps // 2, 2.2, 1.0)
        fraud_probs = base_rate * back_half_boost
    else:
        fraud_probs = np.full(len(eligible_idx), base_rate)

    fraud_draw = rng.random(len(eligible_idx)) < fraud_probs
    fraud_idx = eligible_idx[fraud_draw]
    is_fraud[fraud_idx] = 1

    # Classic PaySim fraud signature: drain the origin account to (near) zero
    # and route a suspiciously large amount, regardless of stated old balance.
    fraud_amounts = rng.lognormal(mean=9.5, sigma=1.0, size=len(fraud_idx))
    if drift:
        # Fraud amounts grow larger in the drifted regime.
        drift_boost = np.where(steps[fraud_idx] > n_steps // 2, 1.6, 1.0)
        fraud_amounts = fraud_amounts * drift_boost

    amounts[fraud_idx] = fraud_amounts
    old_bal_orig[fraud_idx] = np.maximum(fraud_amounts, old_bal_orig[fraud_idx])
    new_bal_orig[fraud_idx] = 0.0
    new_bal_dest[fraud_idx] = old_bal_dest[fraud_idx] + fraud_amounts

    # Simple rule a legacy system might use: flag very large transfers that
    # empty the account. It catches some but not all fraud -- deliberately
    # imperfect, to motivate "why do we need ML at all" in Week 1 discussion.
    is_flagged = np.zeros(n, dtype=int)
    rule_mask = (
        np.isin(types, ["TRANSFER"]) & (amounts > 200_000) & (new_bal_orig == 0)
    )
    is_flagged[rule_mask] = 1

    df = pd.DataFrame(
        {
            "step": steps,
            "type": types,
            "amount": amounts.round(2),
            "nameOrig": orig_ids,
            "oldbalanceOrg": old_bal_orig.round(2),
            "newbalanceOrig": new_bal_orig.round(2),
            "nameDest": dest_ids,
            "oldbalanceDest": old_bal_dest.round(2),
            "newbalanceDest": new_bal_dest.round(2),
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )

    # Deliberately inject missing values into balance columns so students
    # hit a real NaN-handling error rather than a hypothetical one.
    if missing_rate > 0:
        for col in ["oldbalanceDest", "newbalanceDest"]:
            mask = rng.random(n) < missing_rate
            df.loc[mask, col] = np.nan

    df = df.sort_values("step").reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate_transactions()
    print(data.shape)
    print(data["isFraud"].value_counts(normalize=True))
