
# import json
# import math
# import random
# import csv
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import List, Dict, Any

# # -----------------------------
# # Security estimation (conservative table; replace with estimator if available)
# # -----------------------------
# HE_SECURITY_TABLE = {
#     2048: 80,
#     4096: 112,
#     8192: 128,
#     16384: 192,
#     32768: 256,
# }

# def estimate_security_bits(poly_modulus_degree: int, sum_coeff_bits: int) -> int:
#     """
#     Conservative mapping for RLWE security.
#     If you have a lattice estimator, replace this function.
#     Optionally degrade security by unusually large q (sum_coeff_bits).
#     """
#     base = HE_SECURITY_TABLE.get(poly_modulus_degree, 0)
#     # crude degradation if q is very large for a given n (tunable)
#     if poly_modulus_degree == 4096 and sum_coeff_bits > 200:
#         base = min(base, 112)
#     if poly_modulus_degree == 8192 and sum_coeff_bits > 250:
#         base = min(base, 128)
#     if poly_modulus_degree == 16384 and sum_coeff_bits > 350:
#         base = min(base, 192)
#     return base

# # -----------------------------
# # Noise measurement stub
# # -----------------------------
# def measure_final_noise_budget(context: Dict[str, Any]) -> float:
#     """
#     Placeholder noise readout.
#     If you have SEAL/TenSEAL, replace with:
#         decryptor.invariant_noise_budget(ciphertext)
#     This stub correlates with multiplicative depth and sum(q_bits) so ML can learn.
#     """
#     depth = context.get("multiplicative_depth", 1)
#     sum_q = context.get("sum_coeff_bits", 120)
#     n = context.get("poly_modulus_degree", 8192)
#     # higher depth and smaller q -> lower budget (worse)
#     base = 60.0 + 0.02 * (sum_q) - 0.001 * (depth * depth) - (math.log2(n) - 13) * 3.0
#     # add tiny randomness to avoid duplicates
#     noise = max(0.0, base + random.uniform(-1.0, 1.0))
#     return noise

# # -----------------------------
# # Simple synthetic circuit + runtime model (stand-in for your existing code)
# # Replace these with your actual implementations — the point is the output schema.
# # -----------------------------
# @dataclass
# class HardwareProfile:
#     name: str
#     cpu_cores: int
#     cpu_freq_ghz: float
#     memory_gb: int
#     memory_bandwidth_gb_s: float
#     cache_mb: int
#     has_gpu: bool = False

# @dataclass
# class CircuitProfile:
#     circuit_id: str
#     circuit_type: str
#     multiplicative_depth: int
#     operations_count: int
#     rotation_count: int
#     vector_size: int
#     scheme: str = "CKKS"  # or "BFV"

# @dataclass
# class HEParams:
#     poly_modulus_degree: int
#     coeff_mod_bits: List[int]
#     scale_bits: int
#     base: int

# @dataclass
# class RunRecord:
#     # Inputs
#     circuit_id: str
#     circuit_type: str
#     scheme: str
#     poly_modulus_degree: int
#     coeff_mod_bits: str  # JSON string to keep compatibility
#     scale_bits: int
#     base: int
#     multiplicative_depth: int
#     operations_count: int
#     rotation_count: int
#     vector_size: int
#     cpu_cores: int
#     cpu_freq_ghz: float
#     memory_gb: int
#     memory_bandwidth_gb_s: float
#     cache_mb: int
#     has_gpu: int
#     # Derived
#     sum_coeff_bits: int
#     num_primes: int
#     # Labels
#     runtime_s: float
#     memory_mb: float
#     final_noise_budget: float
#     success: int
#     security_bits: int

# def synthetic_runtime_s(params: HEParams, circ: CircuitProfile, hw: HardwareProfile) -> float:
#     n = params.poly_modulus_degree
#     num_primes = len(params.coeff_mod_bits)
#     sum_bits = sum(params.coeff_mod_bits)
#     depth = circ.multiplicative_depth

#     base_cost = (n * math.log2(n) * num_primes) / (hw.cpu_cores * max(hw.cpu_freq_ghz, 1e-3))

#     # make runtime more sensitive to n and q
#     depth_factor = 1.0 + 0.15 * depth
#     q_factor = 1.0 + 0.015 * (sum_bits - 180) / 10.0
#     n_factor = (n / 8192.0) ** 1.15  # stronger dependence on n
#     gpu_boost = 0.5 if hw.has_gpu else 1.0

#     runtime = base_cost * depth_factor * q_factor * n_factor * gpu_boost * 1e-7
#     return runtime


# def synthetic_memory_mb(params: HEParams, circ: CircuitProfile) -> float:
#     n = params.poly_modulus_degree
#     num_primes = len(params.coeff_mod_bits)
#     return max(8.0, (n / 4096) * num_primes * 12.5)

# def generate_coeff_ladders(n: int) -> List[List[int]]:
#     # Sample reasonable ladders under a max bit budget per n
#     budgets = {4096: 218, 8192: 438, 16384: 881, 32768: 1762}
#     budget = budgets.get(n, 438)
#     common_primes = [60, 50, 40, 30, 20]
#     ladders = []
#     for _ in range(6):
#         seq = []
#         total = 0
#         while total < budget - 20 and len(seq) < 6:
#             b = random.choice(common_primes)
#             if total + b <= budget:
#                 seq.append(b)
#                 total += b
#             else:
#                 break
#         if total >= 120:
#             ladders.append(seq)
#     return ladders or [[60, 40, 40, 30]]

# def run_once(circ: CircuitProfile, hw: HardwareProfile, params: HEParams) -> RunRecord:
#     runtime = synthetic_runtime_s(params, circ, hw)
#     mem = synthetic_memory_mb(params, circ)

#     sum_bits = sum(params.coeff_mod_bits)
#     depth = circ.multiplicative_depth
#     n = params.poly_modulus_degree

#     # crude "should decrypt" condition
#     feasible = (sum_bits >= 120 + 8 * depth) and (mem < hw.memory_gb * 1024)

#     # Noise budget (stubbed, replace with library readout)
#     noise_ctx = {
#         "multiplicative_depth": depth,
#         "sum_coeff_bits": sum_bits,
#         "poly_modulus_degree": n,
#     }
#     final_noise = measure_final_noise_budget(noise_ctx)

#     security = estimate_security_bits(n, sum_bits)

#     record = RunRecord(
#         circuit_id=circ.circuit_id,
#         circuit_type=circ.circuit_type,
#         scheme=circ.scheme,
#         poly_modulus_degree=n,
#         coeff_mod_bits=json.dumps(params.coeff_mod_bits),
#         scale_bits=params.scale_bits,
#         base=params.base,
#         multiplicative_depth=depth,
#         operations_count=circ.operations_count,
#         rotation_count=circ.rotation_count,
#         vector_size=circ.vector_size,
#         cpu_cores=hw.cpu_cores,
#         cpu_freq_ghz=hw.cpu_freq_ghz,
#         memory_gb=hw.memory_gb,
#         memory_bandwidth_gb_s=hw.memory_bandwidth_gb_s,
#         cache_mb=hw.cache_mb,
#         has_gpu=1 if hw.has_gpu else 0,
#         sum_coeff_bits=sum_bits,
#         num_primes=len(params.coeff_mod_bits),
#         runtime_s=runtime,
#         memory_mb=mem,
#         final_noise_budget=final_noise,
#         success=1 if feasible else 0,
#         security_bits=security,
#     )
#     return record

# def main_generate(rows_per_combo: int = 3, out_csv: str = "he_parameter_dataset/he_dataset_final.csv"):
#     circuits = [
#         CircuitProfile("c1", "dot", 4, 250, 8, 4096, "CKKS"),
#         CircuitProfile("c2", "matvec", 6, 500, 16, 8192, "CKKS"),
#         CircuitProfile("c3", "poly_eval", 2, 300, 4, 4096, "BFV"),
#         CircuitProfile("c4", "conv1d", 10, 1200, 32, 16384, "CKKS"),
#     ]

#     hws = [
#         HardwareProfile("laptop", 8, 3.2, 16, 25.0, 16, False),
#         HardwareProfile("server", 32, 2.6, 128, 80.0, 64, False),
#         HardwareProfile("gpu_server", 32, 2.6, 128, 80.0, 64, True),
#     ]

#     ns = [4096, 8192, 16384]
#     scale_bits_list = [20, 30, 40]
#     base_list = [2, 3, 5]

#     rows: List[RunRecord] = []

#     for circ in circuits:
#         for hw in hws:
#             for n in ns:
#                 ladders = generate_coeff_ladders(n)
#                 for ladder in ladders:
#                     for sb in scale_bits_list:
#                         for base in base_list:
#                             params = HEParams(n, ladder, sb, base)
#                             for _ in range(rows_per_combo):
#                                 rec = run_once(circ, hw, params)
#                                 rows.append(rec)

#     out_path = Path(out_csv)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     fieldnames = list(asdict(rows[0]).keys())
#     with out_path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for r in rows:
#             writer.writerow(asdict(r))

#     print(f"Wrote {len(rows)} rows to {out_path}")

# if __name__ == "__main__":
#     main_generate(rows_per_combo=3)







# import json
# import math
# import random
# import csv
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import List, Dict, Any

# # -----------------------------
# # Security estimation (conservative table)
# # -----------------------------
# HE_SECURITY_TABLE = {
#     2048: 80,
#     4096: 112,
#     8192: 128,
#     16384: 192,
#     32768: 256,
# }


# def estimate_security_bits(poly_modulus_degree: int, sum_coeff_bits: int) -> int:
#     base = HE_SECURITY_TABLE.get(poly_modulus_degree, 0)
#     if poly_modulus_degree == 4096 and sum_coeff_bits > 200:
#         base = min(base, 112)
#     if poly_modulus_degree == 8192 and sum_coeff_bits > 250:
#         base = min(base, 128)
#     if poly_modulus_degree == 16384 and sum_coeff_bits > 350:
#         base = min(base, 192)
#     return base


# # -----------------------------
# # Noise measurement stub
# # -----------------------------
# def measure_final_noise_budget(context: Dict[str, Any]) -> float:
#     depth = context.get("multiplicative_depth", 1)
#     sum_q = context.get("sum_coeff_bits", 120)
#     n = context.get("poly_modulus_degree", 8192)
#     base = 60.0 + 0.02 * (sum_q) - 0.001 * (depth * depth) - (math.log2(n) - 13) * 3.0
#     noise = max(0.0, base + random.uniform(-1.0, 1.0))
#     return noise


# # -----------------------------
# # Data structures
# # -----------------------------
# @dataclass
# class HardwareProfile:
#     name: str
#     cpu_cores: int
#     cpu_freq_ghz: float
#     memory_gb: int
#     memory_bandwidth_gb_s: float
#     cache_mb: int
#     has_gpu: bool = False


# @dataclass
# class CircuitProfile:
#     circuit_id: str
#     circuit_type: str
#     multiplicative_depth: int
#     operations_count: int
#     rotation_count: int
#     vector_size: int
#     scheme: str = "CKKS"


# @dataclass
# class HEParams:
#     poly_modulus_degree: int
#     coeff_mod_bits: List[int]
#     scale_bits: int
#     base: int


# @dataclass
# class RunRecord:
#     circuit_id: str
#     circuit_type: str
#     scheme: str
#     poly_modulus_degree: int
#     coeff_mod_bits: str
#     scale_bits: int
#     base: int
#     multiplicative_depth: int
#     operations_count: int
#     rotation_count: int
#     vector_size: int
#     cpu_cores: int
#     cpu_freq_ghz: float
#     memory_gb: int
#     memory_bandwidth_gb_s: float
#     cache_mb: int
#     has_gpu: int
#     sum_coeff_bits: int
#     num_primes: int
#     runtime_s: float
#     memory_mb: float
#     final_noise_budget: float
#     success: int
#     security_bits: int


# # -----------------------------
# # TenSEAL-safe helpers
# # -----------------------------
# SEAL_MAX_BITS = {4096: 109, 8192: 218, 16384: 438, 32768: 881}


# def _clamp_ckks_params(poly_mod, coeff_bits, target_scale_bits, depth):
#     coeff = list(coeff_bits)
#     need = (depth + 1)
#     if len(coeff) < need:
#         coeff += [40] * (need - len(coeff))
#     if coeff[-1] < 60:
#         coeff.append(60)

#     limit = SEAL_MAX_BITS.get(int(poly_mod), 218)
#     while sum(coeff) > limit and len(coeff) > 2:
#         coeff.pop(0)
#     if sum(coeff) > limit:
#         coeff = [40, 40, 60] if limit >= 140 else [30, 40]

#     min_prime = min(coeff) if coeff else 40
#     safe_scale_bits = min(target_scale_bits, max(20, min_prime - 2))
#     return coeff, int(safe_scale_bits)


# def _safe_bfv_plain_modulus(poly_mod):
#     return 40961


# # -----------------------------
# # TenSEAL-based runtime
# # -----------------------------
# def synthetic_runtime_s(params: HEParams, circ: CircuitProfile, hw: HardwareProfile) -> float:
#     import time
#     import tenseal as ts

#     scheme = ts.SCHEME_TYPE.CKKS if circ.scheme.upper() == "CKKS" else ts.SCHEME_TYPE.BFV
#     n = int(params.poly_modulus_degree)
#     depth = int(circ.multiplicative_depth)

#     coeff_bits_req = list(params.coeff_mod_bits)
#     scale_bits_req = int(params.scale_bits)

#     if scheme == ts.SCHEME_TYPE.CKKS:
#         coeff_bits, scale_bits = _clamp_ckks_params(n, coeff_bits_req, scale_bits_req, depth)
#         scale = float(2 ** scale_bits)
#     else:
#         coeff_bits = coeff_bits_req[:] if coeff_bits_req else [40, 40, 60]
#         scale = None

#     t0 = time.time()
#     try:
#         if scheme == ts.SCHEME_TYPE.CKKS:
#             ctx = ts.context(
#                 scheme,
#                 poly_modulus_degree=n,
#                 coeff_mod_bit_sizes=coeff_bits,
#             )
#             ctx.global_scale = scale
#         else:
#             ctx = ts.context(
#                 scheme,
#                 poly_modulus_degree=n,
#                 plain_modulus=_safe_bfv_plain_modulus(n),
#                 coeff_mod_bit_sizes=coeff_bits,
#             )

#         vec_len = min(2048, max(64, int(circ.vector_size)))
#         if scheme == ts.SCHEME_TYPE.CKKS:
#             v = [0.1] * vec_len
#             enc = ts.ckks_vector(ctx, v)
#             steps = min(depth, len(coeff_bits) - 1)
#             for _ in range(max(1, steps)):
#                 enc = enc * 1.0009765625
#                 enc = enc + 0.0001
#         else:
#             v = [1] * vec_len
#             enc = ts.bfv_vector(ctx, v)
#             steps = min(depth, 6)
#             for _ in range(max(1, steps)):
#                 enc = enc * 2
#                 enc = enc + 1

#         try:
#             _ = enc.decrypt()
#         except Exception:
#             pass

#         runtime = time.time() - t0

#     except Exception as e:
#         print(f"[Warning] TenSEAL invalid combo (n={n}, q_bits={coeff_bits_req}, scale_bits={scale_bits_req}): {e}")
#         runtime = float("nan")

#     return float(runtime)


# def synthetic_memory_mb(params: HEParams, circ: CircuitProfile) -> float:
#     bytes_per_coeff = 8
#     n = params.poly_modulus_degree
#     num_primes = len(params.coeff_mod_bits)
#     mem_bytes = n * num_primes * bytes_per_coeff * 8
#     return mem_bytes / (1024 * 1024)


# # -----------------------------
# # Dataset generation logic
# # -----------------------------
# def generate_coeff_ladders(n: int) -> List[List[int]]:
#     budgets = {4096: 218, 8192: 438, 16384: 881, 32768: 1762}
#     budget = budgets.get(n, 438)
#     common_primes = [60, 50, 40, 30, 20]
#     ladders = []
#     for _ in range(6):
#         seq = []
#         total = 0
#         while total < budget - 20 and len(seq) < 6:
#             b = random.choice(common_primes)
#             if total + b <= budget:
#                 seq.append(b)
#                 total += b
#             else:
#                 break
#         if total >= 120:
#             ladders.append(seq)
#     return ladders or [[60, 40, 40, 30]]


# def run_once(circ: CircuitProfile, hw: HardwareProfile, params: HEParams) -> RunRecord:
#     runtime = synthetic_runtime_s(params, circ, hw)
#     if math.isnan(runtime):
#         return None

#     mem = synthetic_memory_mb(params, circ)
#     sum_bits = sum(params.coeff_mod_bits)
#     depth = circ.multiplicative_depth
#     n = params.poly_modulus_degree

#     feasible = (sum_bits >= 120 + 8 * depth) and (mem < hw.memory_gb * 1024)

#     noise_ctx = {
#         "multiplicative_depth": depth,
#         "sum_coeff_bits": sum_bits,
#         "poly_modulus_degree": n,
#     }
#     final_noise = measure_final_noise_budget(noise_ctx)
#     security = estimate_security_bits(n, sum_bits)

#     return RunRecord(
#         circuit_id=circ.circuit_id,
#         circuit_type=circ.circuit_type,
#         scheme=circ.scheme,
#         poly_modulus_degree=n,
#         coeff_mod_bits=json.dumps(params.coeff_mod_bits),
#         scale_bits=params.scale_bits,
#         base=params.base,
#         multiplicative_depth=depth,
#         operations_count=circ.operations_count,
#         rotation_count=circ.rotation_count,
#         vector_size=circ.vector_size,
#         cpu_cores=hw.cpu_cores,
#         cpu_freq_ghz=hw.cpu_freq_ghz,
#         memory_gb=hw.memory_gb,
#         memory_bandwidth_gb_s=hw.memory_bandwidth_gb_s,
#         cache_mb=hw.cache_mb,
#         has_gpu=1 if hw.has_gpu else 0,
#         sum_coeff_bits=sum_bits,
#         num_primes=len(params.coeff_mod_bits),
#         runtime_s=runtime,
#         memory_mb=mem,
#         final_noise_budget=final_noise,
#         success=1 if feasible else 0,
#         security_bits=security,
#     )


# # -----------------------------
# # Main driver
# # -----------------------------
# def main_generate(rows_per_combo: int = 3, out_csv: str = "he_parameter_dataset/he_dataset_final.csv"):
#     circuits = [
#         CircuitProfile("c1", "dot", 4, 250, 8, 4096, "CKKS"),
#         CircuitProfile("c2", "matvec", 6, 500, 16, 8192, "CKKS"),
#         CircuitProfile("c3", "poly_eval", 2, 300, 4, 4096, "BFV"),
#         CircuitProfile("c4", "conv1d", 10, 1200, 32, 16384, "CKKS"),
#     ]

#     hws = [
#         HardwareProfile("laptop", 8, 3.2, 16, 25.0, 16, False),
#         HardwareProfile("server", 32, 2.6, 128, 80.0, 64, False),
#         HardwareProfile("gpu_server", 32, 2.6, 128, 80.0, 64, True),
#     ]

#     ns = [4096, 8192, 16384]
#     scale_bits_list = [20, 30, 40]
#     base_list = [2, 3, 5]

#     rows: List[RunRecord] = []

#     for circ in circuits:
#         for hw in hws:
#             for n in ns:
#                 ladders = generate_coeff_ladders(n)
#                 for ladder in ladders:
#                     for sb in scale_bits_list:
#                         for base in base_list:
#                             params = HEParams(n, ladder, sb, base)
#                             for _ in range(rows_per_combo):
#                                 rec = run_once(circ, hw, params)
#                                 if rec is not None:
#                                     rows.append(rec)

#     out_path = Path(out_csv)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if not rows:
#         raise RuntimeError("No valid rows generated!")

#     fieldnames = list(asdict(rows[0]).keys())
#     with out_path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for r in rows:
#             writer.writerow(asdict(r))

#     print(f"Wrote {len(rows)} valid rows to {out_path}")


# if __name__ == "__main__":
#     main_generate(rows_per_combo=3)


import json
import math
import random
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

# -----------------------------
# Security estimation (conservative table)
# -----------------------------
HE_SECURITY_TABLE = {
    2048: 80,
    4096: 112,
    8192: 128,
    16384: 192,
    32768: 256,
}

def estimate_security_bits(poly_modulus_degree: int, sum_coeff_bits: int) -> int:
    base = HE_SECURITY_TABLE.get(poly_modulus_degree, 0)
    if poly_modulus_degree == 4096 and sum_coeff_bits > 200:
        base = min(base, 112)
    if poly_modulus_degree == 8192 and sum_coeff_bits > 250:
        base = min(base, 128)
    if poly_modulus_degree == 16384 and sum_coeff_bits > 350:
        base = min(base, 192)
    return base


# -----------------------------
# Noise measurement stub
# -----------------------------
def measure_final_noise_budget(context: Dict[str, Any]) -> float:
    depth = context.get("multiplicative_depth", 1)
    sum_q = context.get("sum_coeff_bits", 120)
    n = context.get("poly_modulus_degree", 8192)
    base = 60.0 + 0.02 * sum_q - 0.001 * (depth ** 2) - (math.log2(n) - 13) * 3.0
    return max(0.0, base + random.uniform(-1.0, 1.0))


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class HardwareProfile:
    name: str
    cpu_cores: int
    cpu_freq_ghz: float
    memory_gb: int
    memory_bandwidth_gb_s: float
    cache_mb: int
    has_gpu: bool = False


@dataclass
class CircuitProfile:
    circuit_id: str
    circuit_type: str
    multiplicative_depth: int
    operations_count: int
    rotation_count: int
    vector_size: int
    scheme: str = "CKKS"


@dataclass
class HEParams:
    poly_modulus_degree: int
    coeff_mod_bits: List[int]
    scale_bits: int
    base: int


@dataclass
class RunRecord:
    circuit_id: str
    circuit_type: str
    scheme: str
    poly_modulus_degree: int
    coeff_mod_bits: str
    scale_bits: int
    base: int
    multiplicative_depth: int
    operations_count: int
    rotation_count: int
    vector_size: int
    cpu_cores: int
    cpu_freq_ghz: float
    memory_gb: int
    memory_bandwidth_gb_s: float
    cache_mb: int
    has_gpu: int
    sum_coeff_bits: int
    num_primes: int
    runtime_s: float
    memory_mb: float
    final_noise_budget: float
    success: int
    security_bits: int


# -----------------------------
# Global TenSEAL context cache
# -----------------------------
_ctx_cache = {}
SEAL_MAX_BITS = {4096: 109, 8192: 218, 16384: 438, 32768: 881}

def _clamp_ckks_params(poly_mod, coeff_bits, target_scale_bits, depth):
    coeff = list(coeff_bits)
    need = depth + 1
    if len(coeff) < need:
        coeff += [40] * (need - len(coeff))
    if coeff[-1] < 60:
        coeff.append(60)

    limit = SEAL_MAX_BITS.get(int(poly_mod), 218)
    while sum(coeff) > limit and len(coeff) > 2:
        coeff.pop(0)
    if sum(coeff) > limit:
        coeff = [40, 40, 60] if limit >= 140 else [30, 40]

    min_prime = min(coeff) if coeff else 40
    safe_scale_bits = min(target_scale_bits, max(20, min_prime - 2))
    return coeff, int(safe_scale_bits)


def _safe_bfv_plain_modulus(poly_mod):
    return 40961


# -----------------------------
# Cached TenSEAL-based runtime
# -----------------------------
def synthetic_runtime_s(params: HEParams, circ: CircuitProfile, hw: HardwareProfile) -> float:
    import tenseal as ts
    global _ctx_cache

    scheme = ts.SCHEME_TYPE.CKKS if circ.scheme.upper() == "CKKS" else ts.SCHEME_TYPE.BFV
    n = int(params.poly_modulus_degree)
    depth = int(circ.multiplicative_depth)
    key = (n, tuple(params.coeff_mod_bits), params.scale_bits, circ.scheme)

    coeff_bits_req = list(params.coeff_mod_bits)
    scale_bits_req = int(params.scale_bits)

    # Reuse context if already created
    if key not in _ctx_cache:
        try:
            if scheme == ts.SCHEME_TYPE.CKKS:
                coeff_bits, scale_bits = _clamp_ckks_params(n, coeff_bits_req, scale_bits_req, depth)
                ctx = ts.context(
                    scheme,
                    poly_modulus_degree=n,
                    coeff_mod_bit_sizes=coeff_bits,
                )
                ctx.global_scale = float(2 ** scale_bits)
            else:
                coeff_bits = coeff_bits_req[:] if coeff_bits_req else [40, 40, 60]
                ctx = ts.context(
                    scheme,
                    poly_modulus_degree=n,
                    plain_modulus=_safe_bfv_plain_modulus(n),
                    coeff_mod_bit_sizes=coeff_bits,
                )
            _ctx_cache[key] = ctx
        except Exception as e:
            print(f"[Warning] TenSEAL invalid combo (n={n}, q_bits={coeff_bits_req}, scale_bits={scale_bits_req}): {e}")
            return float("nan")

    ctx = _ctx_cache[key]

    # Measure only the encryption and operation time
    vec_len = min(2048, max(64, int(circ.vector_size)))
    start = time.time()

    try:
        if scheme == ts.SCHEME_TYPE.CKKS:
            v = [0.1] * vec_len
            enc = ts.ckks_vector(ctx, v)
            steps = min(depth, 4)
            for _ in range(max(1, steps)):
                enc = enc * 1.001
                enc = enc + 0.001
        else:
            v = [1] * vec_len
            enc = ts.bfv_vector(ctx, v)
            steps = min(depth, 4)
            for _ in range(max(1, steps)):
                enc = enc * 2
                enc = enc + 1
        runtime = time.time() - start
    except Exception:
        runtime = float("nan")

    return float(runtime)


# -----------------------------
# Memory estimation
# -----------------------------
def synthetic_memory_mb(params: HEParams, circ: CircuitProfile) -> float:
    bytes_per_coeff = 8
    n = params.poly_modulus_degree
    num_primes = len(params.coeff_mod_bits)
    mem_bytes = n * num_primes * bytes_per_coeff * 8
    return mem_bytes / (1024 * 1024)


# -----------------------------
# Dataset generation
# -----------------------------
def generate_coeff_ladders(n: int) -> List[List[int]]:
    budgets = {4096: 218, 8192: 438, 16384: 881, 32768: 1762}
    budget = budgets.get(n, 438)
    common_primes = [60, 50, 40, 30, 20]
    ladders = []
    for _ in range(6):
        seq, total = [], 0
        while total < budget - 20 and len(seq) < 6:
            b = random.choice(common_primes)
            if total + b <= budget:
                seq.append(b)
                total += b
            else:
                break
        if total >= 120:
            ladders.append(seq)
    return ladders or [[60, 40, 40, 30]]


def run_once(circ: CircuitProfile, hw: HardwareProfile, params: HEParams) -> RunRecord:
    runtime = synthetic_runtime_s(params, circ, hw)
    if math.isnan(runtime):
        return None

    mem = synthetic_memory_mb(params, circ)
    sum_bits = sum(params.coeff_mod_bits)
    depth = circ.multiplicative_depth
    n = params.poly_modulus_degree

    feasible = (sum_bits >= 120 + 8 * depth) and (mem < hw.memory_gb * 1024)
    noise_ctx = {"multiplicative_depth": depth, "sum_coeff_bits": sum_bits, "poly_modulus_degree": n}
    final_noise = measure_final_noise_budget(noise_ctx)
    security = estimate_security_bits(n, sum_bits)

    return RunRecord(
        circuit_id=circ.circuit_id,
        circuit_type=circ.circuit_type,
        scheme=circ.scheme,
        poly_modulus_degree=n,
        coeff_mod_bits=json.dumps(params.coeff_mod_bits),
        scale_bits=params.scale_bits,
        base=params.base,
        multiplicative_depth=depth,
        operations_count=circ.operations_count,
        rotation_count=circ.rotation_count,
        vector_size=circ.vector_size,
        cpu_cores=hw.cpu_cores,
        cpu_freq_ghz=hw.cpu_freq_ghz,
        memory_gb=hw.memory_gb,
        memory_bandwidth_gb_s=hw.memory_bandwidth_gb_s,
        cache_mb=hw.cache_mb,
        has_gpu=1 if hw.has_gpu else 0,
        sum_coeff_bits=sum_bits,
        num_primes=len(params.coeff_mod_bits),
        runtime_s=runtime,
        memory_mb=mem,
        final_noise_budget=final_noise,
        success=1 if feasible else 0,
        security_bits=security,
    )


def main_generate(rows_per_combo: int = 3, out_csv: str = "he_parameter_dataset/he_dataset_final.csv"):
    circuits = [
        CircuitProfile("c1", "dot", 4, 250, 8, 4096, "CKKS"),
        CircuitProfile("c2", "matvec", 6, 500, 16, 8192, "CKKS"),
        CircuitProfile("c3", "poly_eval", 2, 300, 4, 4096, "BFV"),
        CircuitProfile("c4", "conv1d", 10, 1200, 32, 16384, "CKKS"),
    ]
    hws = [
        HardwareProfile("laptop", 8, 3.2, 16, 25.0, 16, False),
        HardwareProfile("server", 32, 2.6, 128, 80.0, 64, False),
        HardwareProfile("gpu_server", 32, 2.6, 128, 80.0, 64, True),
    ]
    ns = [4096, 8192, 16384]
    scale_bits_list = [20, 30, 40]
    base_list = [2, 3, 5]

    rows: List[RunRecord] = []
    for circ in circuits:
        for hw in hws:
            for n in ns:
                for ladder in generate_coeff_ladders(n):
                    for sb in scale_bits_list:
                        for base in base_list:
                            params = HEParams(n, ladder, sb, base)
                            for _ in range(rows_per_combo):
                                rec = run_once(circ, hw, params)
                                if rec is not None:
                                    rows.append(rec)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No valid rows generated!")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    print(f"Wrote {len(rows)} valid rows to {out_path}")


if __name__ == "__main__":
    main_generate(rows_per_combo=3)
