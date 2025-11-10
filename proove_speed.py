# # import argparse, json, ast, sys
# # from pathlib import Path
# # import numpy as np
# # import pandas as pd

# # # --- import your project code ---
# # from he_data_generator_patched import (
# #     run_once, CircuitProfile, HardwareProfile, HEParams,
# #     estimate_security_bits, synthetic_memory_mb
# # )

# # def parse_args():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--csv", required=True, help="CSV with circuit+hardware profile (first row used)")
# #     ap.add_argument("--rec", default="artifacts/recommended_params.json", help="JSON with ML recommendations")
# #     ap.add_argument("--scale_bits", type=int, default=None, help="Override scale_bits for fair comparison (optional)")
# #     return ap.parse_args()

# # def load_proto(csv_path: str) -> dict:
# #     df = pd.read_csv(csv_path)
# #     if df.empty:
# #         raise SystemExit("CSV is empty.")
# #     return df.iloc[0].to_dict()

# # def build_circuit_hw(proto: dict):
# #     circuit = CircuitProfile(
# #         circuit_id=str(proto.get("circuit_id", "cX")),
# #         circuit_type=str(proto.get("circuit_type", "UNK")),
# #         multiplicative_depth=int(proto.get("multiplicative_depth", 4)),
# #         operations_count=int(proto.get("operations_count", 500)),
# #         rotation_count=int(proto.get("rotation_count", 8)),
# #         vector_size=int(proto.get("vector_size", 4096)),
# #         scheme=str(proto.get("scheme", "CKKS")),
# #     )
# #     hw = HardwareProfile(
# #         name=str(proto.get("hw_name", "proto")),
# #         cpu_cores=int(proto.get("cpu_cores", 8)),
# #         cpu_freq_ghz=float(proto.get("cpu_freq_ghz", 3.0)),
# #         memory_gb=int(proto.get("memory_gb", 16)),
# #         memory_bandwidth_gb_s=float(proto.get("memory_bandwidth_gb_s", 25.0)),
# #         cache_mb=int(proto.get("cache_mb", 16)),
# #         has_gpu=bool(int(proto.get("has_gpu", 0))),
# #     )
# #     return circuit, hw

# # def pick_ml_recommendation(rec_path: str):
# #     p = Path(rec_path)
# #     if not p.exists():
# #         raise SystemExit(f"Missing ML recommendations file: {rec_path}\nRun your recommender first.")
# #     data = json.loads(Path(rec_path).read_text())
# #     if not data:
# #         raise SystemExit("Recommendations JSON is empty.")
# #     top = data[0]
# #     coeff = top["coeff_mod_bits"] if isinstance(top["coeff_mod_bits"], list) else ast.literal_eval(top["coeff_mod_bits"])
# #     return {
# #         "n": int(top["poly_modulus_degree"]),
# #         "coeff_mod_bits": list(map(int, coeff)),
# #         "scale_bits": int(top.get("scale_bits", 30)),
# #         "base": int(top.get("base", 3)),
# #     }

# # def conservative_ladders_16384():
# #     return [
# #         [60,60,50,40],
# #         [60,60,50,40,30],
# #         [60,60,40,40,30],
# #         [50,50,50,40,30],
# #         [60,50,50,40,30],
# #         [60,60,60,50],
# #     ]

# # def eval_config(circuit, hw, n, coeff, scale_bits, base):
# #     params = HEParams(n, coeff, scale_bits, base)
# #     rec = run_once(circuit, hw, params)
# #     return {
# #         "n": n,
# #         "coeff_mod_bits": coeff,
# #         "sum_coeff_bits": int(np.sum(coeff)),
# #         "num_primes": len(coeff),
# #         "scale_bits": scale_bits,
# #         "base": base,
# #         "runtime_s": float(rec.runtime_s),
# #         "success": int(rec.success),
# #         "noise_budget": float(rec.final_noise_budget),
# #         "security_bits": int(rec.security_bits),
# #         "memory_mb": float(rec.memory_mb),
# #     }

# # def main():
# #     args = parse_args()
# #     proto = load_proto(args.csv)
# #     circuit, hw = build_circuit_hw(proto)

# #     # --- ML-picked configuration ---
# #     ml_cfg = pick_ml_recommendation(args.rec)
# #     if args.scale_bits is not None:
# #         ml_cfg["scale_bits"] = int(args.scale_bits)
# #     ml_eval = eval_config(circuit, hw, ml_cfg["n"], ml_cfg["coeff_mod_bits"], ml_cfg["scale_bits"], ml_cfg["base"])

# #     # --- Conservative baseline ---
# #     cons_best = None
# #     for coeff in conservative_ladders_16384():
# #         res = eval_config(circuit, hw, 16384, coeff, ml_cfg["scale_bits"], ml_cfg["base"])
# #         if res["success"] == 1 and res["memory_mb"] < hw.memory_gb * 1024 and res["security_bits"] >= 128:
# #             if cons_best is None or res["runtime_s"] < cons_best["runtime_s"]:
# #                 cons_best = res

# #     if not cons_best:
# #         print("No conservative baseline met constraints.")
# #         sys.exit(1)

# #     # --- Compute detailed precision metrics ---
# #     runtime_ml = ml_eval["runtime_s"]
# #     runtime_cons = cons_best["runtime_s"]
# #     diff = runtime_cons - runtime_ml
# #     ratio = runtime_cons / runtime_ml if runtime_ml > 0 else float("inf")

# #     report = {
# #         "ML_recommended_actual": ml_eval,
# #         "Conservative_baseline_actual": cons_best,
# #         "Detailed_runtime_comparison": {
# #             "runtime_ML_seconds_precise": runtime_ml,
# #             "runtime_Conservative_seconds_precise": runtime_cons,
# #             "runtime_difference_seconds": diff,
# #             "Speedup_factor_conservative_over_ML": ratio
# #         }
# #     }

# #     Path("artifacts").mkdir(parents=True, exist_ok=True)
# #     Path("artifacts/prove_speedup_report.json").write_text(json.dumps(report, indent=2))
# #     print(json.dumps(report, indent=2))
# #     print("\nSaved -> artifacts/prove_speedup_report.json")

# # if __name__ == "__main__":
# #     main()


# import argparse
# import json
# import ast
# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd

# # --- import your project code ---
# from he_data_generator_patched import (
#     run_once, CircuitProfile, HardwareProfile, HEParams,
#     estimate_security_bits, synthetic_memory_mb
# )


# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="CSV with circuit+hardware profile (first row used)")
#     ap.add_argument("--rec", default="artifacts/recommended_params.json", help="JSON with ML recommendations")
#     ap.add_argument("--scale_bits", type=int, default=None, help="Override scale_bits for fair comparison (optional)")
#     return ap.parse_args()


# def load_proto(csv_path: str) -> dict:
#     df = pd.read_csv(csv_path)
#     if df.empty:
#         raise SystemExit("CSV is empty.")
#     return df.iloc[0].to_dict()


# def build_circuit_hw(proto: dict):
#     circuit = CircuitProfile(
#         circuit_id=str(proto.get("circuit_id", "cX")),
#         circuit_type=str(proto.get("circuit_type", "UNK")),
#         multiplicative_depth=int(proto.get("multiplicative_depth", 4)),
#         operations_count=int(proto.get("operations_count", 500)),
#         rotation_count=int(proto.get("rotation_count", 8)),
#         vector_size=int(proto.get("vector_size", 4096)),
#         scheme=str(proto.get("scheme", "CKKS")),
#     )
#     hw = HardwareProfile(
#         name=str(proto.get("hw_name", "proto")),
#         cpu_cores=int(proto.get("cpu_cores", 8)),
#         cpu_freq_ghz=float(proto.get("cpu_freq_ghz", 3.0)),
#         memory_gb=int(proto.get("memory_gb", 16)),
#         memory_bandwidth_gb_s=float(proto.get("memory_bandwidth_gb_s", 25.0)),
#         cache_mb=int(proto.get("cache_mb", 16)),
#         has_gpu=bool(int(proto.get("has_gpu", 0))),
#     )
#     return circuit, hw


# def pick_ml_recommendation(rec_path: str):
#     p = Path(rec_path)
#     if not p.exists():
#         raise SystemExit(f"Missing ML recommendations file: {rec_path}\nRun your recommender first.")
#     data = json.loads(Path(rec_path).read_text())
#     if not data:
#         raise SystemExit("Recommendations JSON is empty.")
#     top = data[0]
#     coeff = top["coeff_mod_bits"] if isinstance(top["coeff_mod_bits"], list) else ast.literal_eval(top["coeff_mod_bits"])
#     return {
#         "n": int(top["poly_modulus_degree"]),
#         "coeff_mod_bits": list(map(int, coeff)),
#         "scale_bits": int(top.get("scale_bits", 30)),
#         "base": int(top.get("base", 3)),
#     }


# def conservative_ladders_16384():
#     return [
#         [60, 60, 50, 40],
#         [60, 60, 50, 40, 30],
#         [60, 60, 40, 40, 30],
#         [50, 50, 50, 40, 30],
#         [60, 50, 50, 40, 30],
#         [60, 60, 60, 50],
#     ]


# def eval_config(circuit, hw, n, coeff, scale_bits, base):
#     params = HEParams(n, coeff, scale_bits, base)
#     rec = run_once(circuit, hw, params)
#     return {
#         "n": n,
#         "coeff_mod_bits": coeff,
#         "sum_coeff_bits": int(np.sum(coeff)),
#         "num_primes": len(coeff),
#         "scale_bits": scale_bits,
#         "base": base,
#         "runtime_s": float(rec.runtime_s),
#         "success": int(rec.success),
#         "noise_budget": float(rec.final_noise_budget),
#         "security_bits": int(rec.security_bits),
#         "memory_mb": float(rec.memory_mb),
#     }


# def main():
#     args = parse_args()
#     proto = load_proto(args.csv)
#     circuit, hw = build_circuit_hw(proto)

#     # --- ML-picked configuration ---
#     ml_cfg = pick_ml_recommendation(args.rec)
#     if args.scale_bits is not None:
#         ml_cfg["scale_bits"] = int(args.scale_bits)

#     ml_eval = eval_config(circuit, hw,
#                           ml_cfg["n"],
#                           ml_cfg["coeff_mod_bits"],
#                           ml_cfg["scale_bits"],
#                           ml_cfg["base"])

#     # --- Conservative baseline ---
#     cons_best = None
#     for coeff in conservative_ladders_16384():
#         res = eval_config(circuit, hw, 16384, coeff, ml_cfg["scale_bits"], ml_cfg["base"])
#         if res["success"] == 1 and res["memory_mb"] < hw.memory_gb * 1024 and res["security_bits"] >= 128:
#             if cons_best is None or res["runtime_s"] < cons_best["runtime_s"]:
#                 cons_best = res

#     if not cons_best:
#         print("No conservative baseline met constraints.")
#         sys.exit(1)

#     # --- Compute performance comparison ---
#     runtime_ml = ml_eval["runtime_s"]
#     runtime_cons = cons_best["runtime_s"]
#     diff = runtime_cons - runtime_ml
#     ratio = runtime_cons / runtime_ml if runtime_ml > 0 else float("inf")

#     report = {
#         "ML_recommended_actual": ml_eval,
#         "Conservative_baseline_actual": cons_best,
#         "Detailed_runtime_comparison": {
#             "runtime_ML_seconds_precise": runtime_ml,
#             "runtime_Conservative_seconds_precise": runtime_cons,
#             "runtime_difference_seconds": diff,
#             "Speedup_factor_conservative_over_ML": ratio
#         }
#     }

#     Path("artifacts").mkdir(parents=True, exist_ok=True)
#     Path("artifacts/prove_speedup_report.json").write_text(json.dumps(report, indent=2))
#     print(json.dumps(report, indent=2))
#     print("\nSaved -> artifacts/prove_speedup_report.json")


# if __name__ == "__main__":
#     main()


# prove_speed.py
# Compare search-time cost:
#   - Classical brute-force (ground-truth run_once on every candidate)
#   - ML surrogate (predict runtime/success, then verify only the winner)
#
# Usage:
#   python prove_speed.py --csv he_parameter_dataset/he_dataset_final.csv \
#       --security-bits 128 --success-thresh 0.98 --mem-headroom 0.8 \
#       --out artifacts/prove_search_speed.json
#
# Notes:
# - Requires trained artifacts: artifacts/runtime_model.pkl, artifacts/success_model.pkl, artifacts/feature_list.json
# - Candidate grid is moderate by default; tweak with flags if needed.

import argparse, ast, json, itertools, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# --- Project imports (ground truth execution) ---
from he_data_generator_patched import (
    run_once, CircuitProfile, HardwareProfile, HEParams,
    estimate_security_bits, synthetic_memory_mb
)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with circuit+hardware profile (first row used)")
    ap.add_argument("--security-bits", type=int, default=128)
    ap.add_argument("--success-thresh", type=float, default=0.98)
    ap.add_argument("--mem-headroom", type=float, default=0.80)
    ap.add_argument("--out", default="artifacts/prove_search_speed.json")
    # Candidate grid controls
    ap.add_argument("--ns", type=str, default="4096,8192,16384", help="Comma-separated n list")
    ap.add_argument("--scales", type=str, default="20,30", help="Comma-separated scale_bits list")
    ap.add_argument("--bases", type=str, default="2,3", help="Comma-separated base list")
    ap.add_argument("--max-ladders-per-n", type=int, default=6, help="Limit ladders per n")
    return ap.parse_args()

# ---------------------------
# Utilities
# ---------------------------
SEC_TABLE = {2048:80, 4096:112, 8192:128, 16384:192, 32768:256}

def security_ok(n, req_bits):
    return SEC_TABLE.get(int(n), 0) >= int(req_bits)

def memory_ok(n, num_primes, hw_mem_gb, headroom=0.8):
    est_mb = max(8.0, (int(n)/4096.0) * int(num_primes) * 12.5)
    return est_mb < (float(hw_mem_gb) * 1024.0 * float(headroom))

def load_proto(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("CSV is empty.")
    return df.iloc[0].to_dict()

def build_circuit_hw(proto: dict):
    circuit = CircuitProfile(
        circuit_id=str(proto.get("circuit_id", "cX")),
        circuit_type=str(proto.get("circuit_type", "UNK")),
        multiplicative_depth=int(proto.get("multiplicative_depth", 4)),
        operations_count=int(proto.get("operations_count", 500)),
        rotation_count=int(proto.get("rotation_count", 8)),
        vector_size=int(proto.get("vector_size", 4096)),
        scheme=str(proto.get("scheme", "CKKS")),
    )
    hw = HardwareProfile(
        name=str(proto.get("hw_name", "proto")),
        cpu_cores=int(proto.get("cpu_cores", 8)),
        cpu_freq_ghz=float(proto.get("cpu_freq_ghz", 3.0)),
        memory_gb=int(proto.get("memory_gb", 16)),
        memory_bandwidth_gb_s=float(proto.get("memory_bandwidth_gb_s", 25.0)),
        cache_mb=int(proto.get("cache_mb", 16)),
        has_gpu=bool(int(proto.get("has_gpu", 0))),
    )
    return circuit, hw

def ladders_for_n(n: int):
    # A compact but meaningful set per n (keep under SEAL bounds & typical CKKS chains)
    presets = {
        4096: [
            [60,40,30], [50,40,40], [50,50,30], [60,30,30], [40,40,40], [50,30,30,20]
        ],
        8192: [
            [60,60,40], [60,50,40], [50,50,40,30], [60,40,40,30], [50,40,40,30], [60,50,30,30]
        ],
        16384: [
            [60,60,50,40], [60,50,40,30], [50,50,40,40], [60,60,40,40], [60,50,50,30]
        ],
    }
    return presets.get(int(n), [[60,60,40], [50,50,40,30]])

def build_candidate_df(proto: dict, ns, scales, bases, max_ladders_per_n: int):
    rows = []
    for n in ns:
        ladders = ladders_for_n(n)[:max(1, max_ladders_per_n)]
        for primes, sb, base in itertools.product(ladders, scales, bases):
            row = proto.copy()
            row["poly_modulus_degree"] = int(n)
            row["coeff_mod_bits"] = json.dumps(primes)
            row["scale_bits"] = int(sb)
            row["base"] = int(base)
            rows.append(row)
    return pd.DataFrame(rows)

# Featureization mirrors training-time logic minimally
def featureize(df: pd.DataFrame, feature_list: dict) -> pd.DataFrame:
    stats = []
    for s in df["coeff_mod_bits"].values:
        primes = ast.literal_eval(s) if isinstance(s, str) else (s or [])
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

    # fill cats
    for col in ["circuit_type", "scheme"]:
        if col not in out.columns:
            out[col] = "UNK"
    # add log2_poly
    poly = pd.to_numeric(out.get("poly_modulus_degree", pd.Series([0]*len(out))), errors="coerce")
    out["log2_poly"] = np.log2(poly).replace([-np.inf, np.inf], np.nan).fillna(0)

    # ensure numeric fields exist
    needed_num = feature_list["numeric"]
    for c in needed_num:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    if "has_gpu" in out.columns:
        out["has_gpu"] = (pd.to_numeric(out["has_gpu"], errors="coerce").fillna(0).astype(float) > 0).astype(int)

    return out

def eval_ground_truth(circuit, hw, n, coeff, scale_bits, base):
    params = HEParams(int(n), list(map(int, coeff)), int(scale_bits), int(base))
    rec = run_once(circuit, hw, params)
    return None if rec is None else {
        "n": int(n),
        "coeff_mod_bits": list(map(int, coeff)),
        "sum_coeff_bits": int(np.sum(coeff)),
        "num_primes": len(coeff),
        "scale_bits": int(scale_bits),
        "base": int(base),
        "runtime_s": float(rec.runtime_s),
        "success": int(rec.success),
        "noise_budget": float(rec.final_noise_budget),
        "security_bits": int(rec.security_bits),
        "memory_mb": float(rec.memory_mb),
    }

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    scales = [int(x) for x in args.scales.split(",") if x.strip()]
    bases = [int(x) for x in args.bases.split(",") if x.strip()]

    proto = load_proto(args.csv)
    circuit, hw = build_circuit_hw(proto)

    # Candidate grid
    cand_df = build_candidate_df(proto, ns, scales, bases, args.max_ladders_per_n)
    num_candidates = len(cand_df)

    # -----------------------
    # Classical brute-force
    # -----------------------
    classical_best = None
    classical_evaluated = 0
    t0 = time.perf_counter()
    for _, row in cand_df.iterrows():
        coeff = ast.literal_eval(row["coeff_mod_bits"])
        n = int(row["poly_modulus_degree"])
        if not security_ok(n, args.security_bits):
            continue
        # memory check (cheap)
        if not memory_ok(n, len(coeff), hw.memory_gb, args.mem_headroom):
            continue
        gt = eval_ground_truth(circuit, hw, n, coeff, row["scale_bits"], row["base"])
        classical_evaluated += 1
        if gt and (gt["success"] == 1) and (gt["security_bits"] >= args.security_bits) and \
           (gt["memory_mb"] < hw.memory_gb * 1024 * args.mem_headroom):
            if (classical_best is None) or (gt["runtime_s"] < classical_best["runtime_s"]):
                classical_best = gt
    classical_time = time.perf_counter() - t0

    # -----------------------
    # ML surrogate search
    # -----------------------
    ml_report = {
        "loaded_runtime_model": False,
        "loaded_success_model": False
    }
    try:
        runtime_model = joblib.load("artifacts/runtime_model.pkl")
        ml_report["loaded_runtime_model"] = True
    except Exception:
        runtime_model = None

    try:
        success_model = joblib.load("artifacts/success_model.pkl")
        ml_report["loaded_success_model"] = True
    except Exception:
        success_model = None

    feature_list = None
    try:
        feature_list = json.loads(Path("artifacts/feature_list.json").read_text())
    except Exception:
        # minimal fallback if missing
        feature_list = {
            "categorical": ["circuit_type", "scheme"],
            "numeric": [
                "poly_modulus_degree",
                "multiplicative_depth","operations_count","rotation_count","vector_size",
                "cpu_cores","cpu_freq_ghz","memory_gb","memory_bandwidth_gb_s","cache_mb","has_gpu",
                "num_primes","sum_coeff_bits",
                "max_prime_bits","min_prime_bits","mean_prime_bits","last_prime_bits",
                "log2_poly",
            ],
        }

    # Featureize once for all candidates
    feat = featureize(cand_df.copy(), feature_list)
    cols = feature_list["categorical"] + feature_list["numeric"]
    X = feat[cols]

    t1 = time.perf_counter()
    # Default predictions if models missing
    pred_runtime = np.full((len(feat),), np.inf, dtype=float)
    pred_success = np.zeros((len(feat),), dtype=float)

    if runtime_model is not None:
        pred_runtime = runtime_model.predict(X)
    # Success model may be a calibrated wrapper; handle gracefully
    if success_model is not None:
        try:
            pred_success = success_model.predict_proba(X)[:, 1]
        except Exception:
            # If not calibrated/fitted, fallback to zeros
            pred_success = np.zeros((len(feat),), dtype=float)

    feat["pred_runtime_s"] = pred_runtime
    feat["pred_success"] = pred_success

    # Apply constraints cheaply (security, success, memory)
    hw_mem_gb = float(proto.get("memory_gb", 16))
    feas = []
    for i, row in feat.iterrows():
        n = int(row["poly_modulus_degree"])
        # num_primes/sum_coeff_bits were computed in featureize
        pred_ok = True
        if not security_ok(n, args.security_bits):
            pred_ok = False
        elif row["pred_success"] < args.success_thresh:
            pred_ok = False
        elif not memory_ok(n, int(row["num_primes"]), hw_mem_gb, args.mem_headroom):
            pred_ok = False
        feas.append(pred_ok)
    feat["feasible"] = feas

    # Pick ML winner (fast)
    feas_df = feat[feat["feasible"]].copy()
    if feas_df.empty:
        feas_df = feat.copy()  # relax if none feasible under ML predicate
    ml_pick_row = feas_df.sort_values(["pred_runtime_s","poly_modulus_degree","sum_coeff_bits"]).iloc[0]

    ml_time_search = time.perf_counter() - t1

    # Verify ML pick with a single ground-truth run
    ml_coeff = ast.literal_eval(ml_pick_row["coeff_mod_bits"])
    ml_gt = eval_ground_truth(
        circuit, hw,
        n=int(ml_pick_row["poly_modulus_degree"]),
        coeff=ml_coeff,
        scale_bits=int(ml_pick_row["scale_bits"]),
        base=int(ml_pick_row["base"])
    )

    # -----------------------
    # Report
    # -----------------------
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    report = {
        "meta": {
            "num_candidates": num_candidates,
            "grid_ns": ns,
            "grid_scales": scales,
            "grid_bases": bases,
            "max_ladders_per_n": args.max_ladders_per_n,
        },
        "Classical_search": {
            "time_seconds": classical_time,
            "candidates_evaluated": classical_evaluated,
            "best_actual": classical_best,
        },
        "ML_search": {
            "time_seconds": ml_time_search,
            "models_loaded": ml_report,
            "candidates_scored": int(len(feat)),
            "winner_pred": {
                "poly_modulus_degree": int(ml_pick_row["poly_modulus_degree"]),
                "coeff_mod_bits": ast.literal_eval(ml_pick_row["coeff_mod_bits"]),
                "scale_bits": int(ml_pick_row["scale_bits"]),
                "base": int(ml_pick_row["base"]),
                "pred_runtime_s": float(ml_pick_row["pred_runtime_s"]),
                "pred_success": float(ml_pick_row["pred_success"]),
                "num_primes": int(ml_pick_row["num_primes"]),
                "sum_coeff_bits": int(ml_pick_row["sum_coeff_bits"]),
            },
            "winner_actual_verified": ml_gt,
        },
        "Search_time_speedup_classical_over_ML":
            (classical_time / ml_time_search) if ml_time_search > 0 else float("inf"),
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()
