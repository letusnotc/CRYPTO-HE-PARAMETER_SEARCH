"""
Feature engineering + model training for:
- Runtime regressor
- Success classifier
- (Optional) Noise regressor if final_noise_budget present

Usage:
    python ml_feature_and_train.py --csv he_parameter_dataset/he_dataset_final.csv
Outputs:
    artifacts/runtime_model.pkl
    artifacts/success_model.pkl
    artifacts/noise_model.pkl (if label present)
    artifacts/feature_list.json
    artifacts/metrics.json
"""

import argparse
import json
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Silence warnings
warnings.filterwarnings("ignore")

# Try LightGBM; fallback to RandomForest if not available
USE_LGBM = True
try:
    import lightgbm as lgb
    lgb.set_config(verbosity=-1)
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    USE_LGBM = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to he runs csv")
    return ap.parse_args()


def extract_coeff_features(row):
    try:
        primes = ast.literal_eval(row["coeff_mod_bits"])
    except Exception:
        primes = []
    if not primes:
        return pd.Series({
            "max_prime_bits": 0,
            "min_prime_bits": 0,
            "mean_prime_bits": 0.0,
            "last_prime_bits": 0
        })
    return pd.Series({
        "max_prime_bits": int(np.max(primes)),
        "min_prime_bits": int(np.min(primes)),
        "mean_prime_bits": float(np.mean(primes)),
        "last_prime_bits": int(primes[-1]),
    })


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    coeff = df.apply(extract_coeff_features, axis=1)
    out = pd.concat([df.copy(), coeff], axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]

    for col in ["circuit_type", "scheme"]:
        if col not in out.columns:
            out[col] = "UNK"

    poly = pd.to_numeric(out.get("poly_modulus_degree", pd.Series([0] * len(out))), errors="coerce")
    out["log2_poly"] = np.log2(poly).replace([-np.inf, np.inf], np.nan).fillna(0)

    num_fill_cols = [
        "poly_modulus_degree",
        "multiplicative_depth", "operations_count", "rotation_count", "vector_size",
        "cpu_cores", "cpu_freq_ghz", "memory_gb", "memory_bandwidth_gb_s", "cache_mb", "has_gpu",
        "num_primes", "sum_coeff_bits",
        "max_prime_bits", "min_prime_bits", "mean_prime_bits", "last_prime_bits",
        "log2_poly",
    ]

    for c in num_fill_cols:
        if c not in out.columns:
            out[c] = 0

    def _as_series(obj, length):
        from pandas import Series
        return obj if isinstance(obj, Series) else Series([obj] * length)

    for c in num_fill_cols:
        col = out[c]
        col = _as_series(col, len(out))
        col = pd.to_numeric(col, errors="coerce").fillna(0)
        out.loc[:, c] = col.values

    out["has_gpu"] = (out["has_gpu"].astype(float) > 0).astype(int)
    return out


def get_feature_space(df: pd.DataFrame):
    cat_cols = ["circuit_type", "scheme"]
    num_cols = [
        "poly_modulus_degree",
        "multiplicative_depth", "operations_count", "rotation_count", "vector_size",
        "cpu_cores", "cpu_freq_ghz", "memory_gb", "memory_bandwidth_gb_s", "cache_mb", "has_gpu",
        "num_primes", "sum_coeff_bits",
        "max_prime_bits", "min_prime_bits", "mean_prime_bits", "last_prime_bits",
        "log2_poly",
    ]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    feature_list = {"categorical": cat_cols, "numeric": num_cols}
    return pre, feature_list


def train_runtime(df_feat: pd.DataFrame, pre, feature_list):
    df_local = df_feat[df_feat["runtime_s"].notnull()].copy()
    y = df_local["runtime_s"].astype(float).values
    X = df_local[feature_list["categorical"] + feature_list["numeric"]]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    if USE_LGBM:
        model = Pipeline([
            ("pre", pre),
            ("lgbm", LGBMRegressor(
                n_estimators=1500, learning_rate=0.03, subsample=0.9, verbosity=-1
            ))
        ])
    else:
        model = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=400, random_state=42))
        ])

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    rmae = float(mae / max(1e-9, np.mean(yte)))
    joblib.dump(model, ARTIFACTS / "runtime_model.pkl")
    return {"runtime_mae": mae, "runtime_rmse": rmse, "runtime_rmae": rmae}


def train_success(df_feat: pd.DataFrame, pre, feature_list):
    if "success" not in df_feat.columns:
        return {"success_auc": None, "success_brier": None}

    y = df_feat["success"].astype(int).values
    X = df_feat[feature_list["categorical"] + feature_list["numeric"]]

    # If only one class exists (e.g. all 1s or all 0s), skip AUC/Brier
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"[Warning] Only one class ({unique_classes[0]}) found in 'success'; skipping AUC/Brier.")
        # Train a dummy model anyway for completeness
        if USE_LGBM:
            base = Pipeline([
                ("pre", pre),
                ("lgbm", LGBMClassifier(n_estimators=10))
            ])
        else:
            base = Pipeline([
                ("pre", pre),
                ("rf", RandomForestClassifier(n_estimators=10))
            ])
        model = CalibratedClassifierCV(base, cv="prefit")
        joblib.dump(model, ARTIFACTS / "success_model.pkl")
        return {"success_auc": None, "success_brier": None}

    # Train/test split (with stratification since multiple classes exist)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)

    if USE_LGBM:
        base = Pipeline([
            ("pre", pre),
            ("lgbm", LGBMClassifier(
                n_estimators=1500, learning_rate=0.03, subsample=0.9, verbosity=-1
            ))
        ])
        model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    else:
        base = Pipeline([
            ("pre", pre),
            ("rf", RandomForestClassifier(n_estimators=600, random_state=24))
        ])
        model = CalibratedClassifierCV(base, cv=3, method="isotonic")

    model.fit(Xtr, ytr)

    # Predict probabilities safely
    prob = model.predict_proba(Xte)
    if prob.shape[1] == 1:  # only one column -> single class
        print("[Warning] Only one probability column returned; skipping metrics.")
        auc, brier = None, None
    else:
        prob = prob[:, 1]
        try:
            auc = float(roc_auc_score(yte, prob))
        except Exception:
            auc = None
        try:
            brier = float(brier_score_loss(yte, prob))
        except Exception:
            brier = None

    joblib.dump(model, ARTIFACTS / "success_model.pkl")
    return {"success_auc": auc, "success_brier": brier}



def train_noise(df_feat: pd.DataFrame, pre, feature_list):
    if "final_noise_budget" not in df_feat.columns:
        return {"noise_mae": None, "noise_rmse": None, "noise_rmae": None}

    y = pd.to_numeric(df_feat["final_noise_budget"], errors="coerce")
    mask = y.notnull()
    y = y[mask].astype(float).values
    X = df_feat.loc[mask, feature_list["categorical"] + feature_list["numeric"]]

    if len(y) < 20:
        return {"noise_mae": None, "noise_rmse": None, "noise_rmae": None}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7)

    if USE_LGBM:
        model = Pipeline([
            ("pre", pre),
            ("lgbm", LGBMRegressor(
                n_estimators=1500, learning_rate=0.03, subsample=0.9, verbosity=-1
            ))
        ])
    else:
        model = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=400, random_state=7))
        ])

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    rmae = float(mae / max(1e-9, np.mean(yte)))
    joblib.dump(model, ARTIFACTS / "noise_model.pkl")
    return {"noise_mae": mae, "noise_rmse": rmse, "noise_rmae": rmae}


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df_feat = build_features(df)
    pre, feature_list = get_feature_space(df_feat)

    with open(ARTIFACTS / "feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)

    metrics = {}
    metrics.update(train_runtime(df_feat, pre, feature_list))
    metrics.update(train_success(df_feat, pre, feature_list))
    metrics.update(train_noise(df_feat, pre, feature_list))

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
