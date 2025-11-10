# """
# Parameter recommendation using trained artifacts.

# Adds:
# - Expanded candidate ladders
# - Security + Success + Memory constraints
# - JSON export

# Usage:
#   python recommend_params_demo.py --csv he_parameter_dataset/he_dataset_final.csv --topk 5 ^
#     --success-thresh 0.98 --security-bits 128 --mem-headroom 0.8
# """

# import argparse, json, ast, itertools
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import joblib


# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="CSV to copy circuit+hardware profile from")
#     ap.add_argument("--topk", type=int, default=3)
#     ap.add_argument("--success-thresh", type=float, default=0.98, help="Min P(success)")
#     ap.add_argument("--security-bits", type=int, default=128, help="Min security bits")
#     ap.add_argument("--mem-headroom", type=float, default=0.80,
#                     help="Max memory % to allow (e.g., 0.8 means 80% of HW mem)")
#     ap.add_argument("--out", default="artifacts/recommended_params.json", help="JSON output file")
#     return ap.parse_args()


# def load_feature_list():
#     with open("artifacts/feature_list.json","r") as f:
#         return json.load(f)

# def security_ok(n, required_bits=128):
#     table = {2048:80, 4096:112, 8192:128, 16384:192, 32768:256}
#     return table.get(int(n), 0) >= required_bits

# def memory_ok(n, num_primes, hw_mem_gb, headroom=0.8):
#     est_mb = max(8.0, (int(n)/4096.0) * int(num_primes) * 12.5)
#     return est_mb < (float(hw_mem_gb) * 1024.0 * float(headroom))

# def featureize(df, feature_list):
#     # Parse coeff_mod_bits -> stats used in training
#     stats = []
#     for s in df["coeff_mod_bits"].values:
#         primes = ast.literal_eval(s)
#         if not primes:
#             stats.append({
#                 "num_primes": 0, "sum_coeff_bits": 0, "max_prime_bits": 0,
#                 "min_prime_bits": 0, "mean_prime_bits": 0.0, "last_prime_bits": 0
#             })
#         else:
#             stats.append({
#                 "num_primes": len(primes),
#                 "sum_coeff_bits": int(np.sum(primes)),
#                 "max_prime_bits": int(np.max(primes)),
#                 "min_prime_bits": int(np.min(primes)),
#                 "mean_prime_bits": float(np.mean(primes)),
#                 "last_prime_bits": int(primes[-1]),
#             })
#     f = pd.DataFrame(stats, index=df.index)
#     out = pd.concat([df.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
#     out = out.loc[:, ~out.columns.duplicated(keep="last")]

#     for col in ["circuit_type", "scheme"]:
#         if col not in out.columns:
#             out[col] = "UNK"

#     poly = pd.to_numeric(out.get("poly_modulus_degree", pd.Series([0]*len(out))), errors="coerce")
#     out["log2_poly"] = np.log2(poly).replace([-np.inf, np.inf], np.nan).fillna(0)

#     needed_num = feature_list["numeric"]
#     n = len(out)
#     for c in needed_num:
#         if c not in out.columns:
#             out[c] = 0

#     num_block = {}
#     for c in needed_num:
#         col = out[c]
#         if not isinstance(col, pd.Series) or len(col) != n:
#             col = pd.Series([col] * n, index=out.index)
#         num_block[c] = pd.to_numeric(col, errors="coerce").fillna(0).values
#     out[needed_num] = pd.DataFrame(num_block, index=out.index)

#     if "has_gpu" in out.columns:
#         out["has_gpu"] = (pd.to_numeric(out["has_gpu"], errors="coerce").fillna(0).astype(float) > 0).astype(int)

#     return out


# def build_ladders(n):
#     base_sets = {
#         4096: [
#             [60,40,30], [50,40,40], [60,50,30], [50,50,30], [40,40,40],
#             [60,30,30], [50,30,30,20]
#         ],
#         8192: [
#             [60,60,40], [60,50,40], [50,50,40,30], [60,40,40,30], [50,40,40,30],
#             [60,50,30,30], [50,50,30,30]
#         ],
#         16384: [
#             [60,60,50,40], [60,50,40,30], [50,50,40,40], [60,60,40,40], [60,50,50,30]
#         ]
#     }
#     return base_sets.get(n, [[60,60,40], [50,50,40,30]])

# def build_candidates(proto_row):
#     ns = [4096, 8192, 16384]
#     scale_bits_list = [20, 30, 40]
#     base_list = [2, 3, 5]
#     rows = []
#     for n in ns:
#         for primes, sb, base in itertools.product(build_ladders(n), scale_bits_list, base_list):
#             row = proto_row.copy()
#             row["poly_modulus_degree"] = n
#             row["coeff_mod_bits"] = json.dumps(primes)
#             row["scale_bits"] = sb
#             row["base"] = base
#             rows.append(row)
#     return pd.DataFrame(rows)


# def main():
#     args = parse_args()

#     df = pd.read_csv(args.csv)
#     if df.empty:
#         raise SystemExit("CSV is empty.")
#     proto = df.iloc[0].to_dict()

#     cand = build_candidates(proto)
#     feature_list = load_feature_list()
#     feat = featureize(cand, feature_list)

#     for col in feature_list["categorical"]:
#         if col not in feat.columns:
#             feat[col] = "UNK"
#     for col in feature_list["numeric"]:
#         if col not in feat.columns:
#             feat[col] = 0
#     cols = feature_list["categorical"] + feature_list["numeric"]
#     X = feat[cols]

#     runtime_model = joblib.load("artifacts/runtime_model.pkl")
#     success_model = joblib.load("artifacts/success_model.pkl")

#     feat["pred_runtime_s"] = runtime_model.predict(X)
#     feat["pred_success"] = success_model.predict_proba(X)[:, 1]

#     hw_mem_gb = float(proto.get("memory_gb", 16))
#     ok = []
#     for _, row in feat.iterrows():
#         n = int(row["poly_modulus_degree"])
#         num_primes = int(row["num_primes"])
#         if (security_ok(n, args.security_bits)
#             and row["pred_success"] >= args.success_thresh
#             and memory_ok(n, num_primes, hw_mem_gb, args.mem_headroom)):
#             ok.append(True)
#         else:
#             ok.append(False)
#     feat["feasible"] = ok

#     feasible = feat[feat["feasible"]].copy()
#     if feasible.empty:
#         print("No feasible candidates under current thresholds. Showing fastest overall instead.")
#         feasible = feat.copy()

#     feasible = feasible.sort_values(
#         ["pred_runtime_s","poly_modulus_degree","sum_coeff_bits"]
#     ).head(args.topk)

#     out_cols = ["poly_modulus_degree","coeff_mod_bits","scale_bits","base",
#                 "pred_runtime_s","pred_success","num_primes","sum_coeff_bits"]
#     print(feasible[out_cols].to_string(index=False))

#     Path("artifacts").mkdir(parents=True, exist_ok=True)
#     recs = feasible[out_cols].copy()
#     recs["coeff_mod_bits"] = recs["coeff_mod_bits"].apply(lambda s: ast.literal_eval(s))
#     Path(args.out).write_text(json.dumps(recs.to_dict(orient="records"), indent=2))
#     print(f"\nSaved -> {args.out}")


# if __name__ == "__main__":
#     main()


"""
Parameter recommendation using trained artifacts.

Adds:
- Expanded candidate ladders
- Security + Success + Memory constraints
- JSON export

Usage:
  python recommend_params_demo.py --csv he_parameter_dataset/he_dataset_final.csv --topk 5 ^
    --success-thresh 0.98 --security-bits 128 --mem-headroom 0.8
"""

import argparse, json, ast, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV to copy circuit+hardware profile from")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--success-thresh", type=float, default=0.98, help="Min P(success)")
    ap.add_argument("--security-bits", type=int, default=128, help="Min security bits")
    ap.add_argument("--mem-headroom", type=float, default=0.80,
                    help="Max memory % to allow (e.g., 0.8 means 80% of HW mem)")
    ap.add_argument("--out", default="artifacts/recommended_params.json", help="JSON output file")
    return ap.parse_args()


def load_feature_list():
    with open("artifacts/feature_list.json", "r") as f:
        return json.load(f)


def security_ok(n, required_bits=128):
    table = {2048: 80, 4096: 112, 8192: 128, 16384: 192, 32768: 256}
    return table.get(int(n), 0) >= required_bits


def memory_ok(n, num_primes, hw_mem_gb, headroom=0.8):
    est_mb = max(8.0, (int(n) / 4096.0) * int(num_primes) * 12.5)
    return est_mb < (float(hw_mem_gb) * 1024.0 * float(headroom))


def featureize(df, feature_list):
    # Parse coeff_mod_bits -> stats used in training
    stats = []
    for s in df["coeff_mod_bits"].values:
        primes = ast.literal_eval(s)
        if not primes:
            stats.append({
                "num_primes": 0, "sum_coeff_bits": 0, "max_prime_bits": 0,
                "min_prime_bits": 0, "mean_prime_bits": 0.0, "last_prime_bits": 0
            })
        else:
            stats.append({
                "num_primes": len(primes),
                "sum_coeff_bits": int(np.sum(primes)),
                "max_prime_bits": int(np.max(primes)),
                "min_prime_bits": int(np.min(primes)),
                "mean_prime_bits": float(np.mean(primes)),
                "last_prime_bits": int(primes[-1]),
            })
    f = pd.DataFrame(stats, index=df.index)
    out = pd.concat([df.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="last")]

    for col in ["circuit_type", "scheme"]:
        if col not in out.columns:
            out[col] = "UNK"

    poly = pd.to_numeric(out.get("poly_modulus_degree", pd.Series([0] * len(out))), errors="coerce")
    out["log2_poly"] = np.log2(poly).replace([-np.inf, np.inf], np.nan).fillna(0)

    needed_num = feature_list["numeric"]
    n = len(out)
    for c in needed_num:
        if c not in out.columns:
            out[c] = 0

    num_block = {}
    for c in needed_num:
        col = out[c]
        if not isinstance(col, pd.Series) or len(col) != n:
            col = pd.Series([col] * n, index=out.index)
        num_block[c] = pd.to_numeric(col, errors="coerce").fillna(0).values
    out[needed_num] = pd.DataFrame(num_block, index=out.index)

    if "has_gpu" in out.columns:
        out["has_gpu"] = (pd.to_numeric(out["has_gpu"], errors="coerce").fillna(0).astype(float) > 0).astype(int)

    return out


def build_ladders(n):
    base_sets = {
        4096: [
            [60, 40, 30], [50, 40, 40], [60, 50, 30], [50, 50, 30], [40, 40, 40],
            [60, 30, 30], [50, 30, 30, 20]
        ],
        8192: [
            [60, 60, 40], [60, 50, 40], [50, 50, 40, 30], [60, 40, 40, 30],
            [50, 40, 40, 30], [60, 50, 30, 30], [50, 50, 30, 30]
        ],
        16384: [
            [60, 60, 50, 40], [60, 50, 40, 30], [50, 50, 40, 40], [60, 60, 40, 40],
            [60, 50, 50, 30]
        ]
    }
    return base_sets.get(n, [[60, 60, 40], [50, 50, 40, 30]])


def build_candidates(proto_row):
    ns = [4096, 8192, 16384]
    scale_bits_list = [20, 30, 40]
    base_list = [2, 3, 5]
    rows = []
    for n in ns:
        for primes, sb, base in itertools.product(build_ladders(n), scale_bits_list, base_list):
            row = proto_row.copy()
            row["poly_modulus_degree"] = n
            row["coeff_mod_bits"] = json.dumps(primes)
            row["scale_bits"] = sb
            row["base"] = base
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("CSV is empty.")
    proto = df.iloc[0].to_dict()

    cand = build_candidates(proto)
    feature_list = load_feature_list()
    feat = featureize(cand, feature_list)

    for col in feature_list["categorical"]:
        if col not in feat.columns:
            feat[col] = "UNK"
    for col in feature_list["numeric"]:
        if col not in feat.columns:
            feat[col] = 0
    cols = feature_list["categorical"] + feature_list["numeric"]
    X = feat[cols]

    # Load models
    runtime_model = joblib.load("artifacts/runtime_model.pkl")
    success_model = joblib.load("artifacts/success_model.pkl")

    feat["pred_runtime_s"] = runtime_model.predict(X)

    # --- Safe success prediction handling ---
    try:
        pred_success = success_model.predict_proba(X)
        if pred_success.shape[1] == 1:
            print("[⚠️ Warning] success_model has only one probability column; assuming all successes.")
            feat["pred_success"] = 1.0
        else:
            feat["pred_success"] = pred_success[:, 1]
    except Exception as e:
        print(f"[⚠️ Warning] success_model not fitted or unusable ({e}); assuming success=1.0 for all candidates.")
        feat["pred_success"] = 1.0

    # --- Apply feasibility filters ---
    hw_mem_gb = float(proto.get("memory_gb", 16))
    ok = []
    for _, row in feat.iterrows():
        n = int(row["poly_modulus_degree"])
        num_primes = int(row["num_primes"])
        if (security_ok(n, args.security_bits)
            and row["pred_success"] >= args.success_thresh
            and memory_ok(n, num_primes, hw_mem_gb, args.mem_headroom)):
            ok.append(True)
        else:
            ok.append(False)
    feat["feasible"] = ok

    feasible = feat[feat["feasible"]].copy()
    if feasible.empty:
        print("No feasible candidates under current thresholds. Showing fastest overall instead.")
        feasible = feat.copy()

    feasible = feasible.sort_values(
        ["pred_runtime_s", "poly_modulus_degree", "sum_coeff_bits"]
    ).head(args.topk)

    out_cols = ["poly_modulus_degree", "coeff_mod_bits", "scale_bits", "base",
                "pred_runtime_s", "pred_success", "num_primes", "sum_coeff_bits"]
    print(feasible[out_cols].to_string(index=False))

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    recs = feasible[out_cols].copy()
    recs["coeff_mod_bits"] = recs["coeff_mod_bits"].apply(lambda s: ast.literal_eval(s))
    Path(args.out).write_text(json.dumps(recs.to_dict(orient="records"), indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
