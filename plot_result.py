# import json
# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# def main():
#     json_path = Path("artifacts/prove_speedup_report.json")
#     if not json_path.exists():
#         print(f"❌ File not found: {json_path}")
#         print("Run `prove_speedup.py` first to generate it.")
#         return

#     # Load JSON report
#     data = json.loads(json_path.read_text())

#     ml = data["ML_recommended_actual"]
#     cons = data["Conservative_baseline_actual"]
#     compare = data["Detailed_runtime_comparison"]

#     # Extract values
#     models = ["ML Recommended", "Conservative Baseline"]
#     runtime = [ml["runtime_s"], cons["runtime_s"]]
#     memory = [ml["memory_mb"], cons["memory_mb"]]
#     security = [ml["security_bits"], cons["security_bits"]]

#     # Create figure
#     plt.figure(figsize=(10, 5))
#     bar_width = 0.35
#     indices = np.arange(len(models))

#     # --- Runtime Plot ---
#     plt.subplot(1, 2, 1)
#     plt.bar(indices, runtime, color=["#4CAF50", "#2196F3"], width=bar_width)
#     plt.title("Runtime Comparison", fontsize=14)
#     plt.xticks(indices, models)
#     plt.ylabel("Runtime (seconds)")
#     plt.text(0, runtime[0] + 0.0005, f"{runtime[0]:.4f}s", ha='center')
#     plt.text(1, runtime[1] + 0.0005, f"{runtime[1]:.4f}s", ha='center')

#     # --- Memory Plot ---
#     plt.subplot(1, 2, 2)
#     plt.bar(indices, memory, color=["#8BC34A", "#03A9F4"], width=bar_width)
#     plt.title("Memory Usage Comparison", fontsize=14)
#     plt.xticks(indices, models)
#     plt.ylabel("Memory (MB)")
#     plt.text(0, memory[0] + 5, f"{memory[0]:.1f} MB", ha='center')
#     plt.text(1, memory[1] + 5, f"{memory[1]:.1f} MB", ha='center')

#     plt.suptitle("HE Parameter Optimization: ML vs Conservative", fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.95])

#     # Save plot
#     out_path = Path("artifacts/plot_result.png")
#     plt.savefig(out_path, dpi=300)
#     plt.show()

#     # Print summary
#     print("\n✅ Plot saved at:", out_path)
#     print("📊 Speedup Factor:", round(compare["Speedup_factor_conservative_over_ML"], 2))
#     print("🧠 Memory Reduction:", f"{memory[1]/memory[0]:.2f}× smaller in ML config")
#     print("🔐 Security: ML =", ml['security_bits'], "bits | Conservative =", cons['security_bits'], "bits")

# if __name__ == "__main__":
#     main()


import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    json_path = Path("artifacts/prove_search_speed.json")
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print("Run `prove_speed.py` first to generate it.")
        return

    # --- Load JSON ---
    data = json.loads(json_path.read_text())

    classical = data["Classical_search"]
    ml = data["ML_search"]
    speedup = data["Search_time_speedup_classical_over_ML"]

    # --- Extract search times ---
    search_labels = ["Classical (Brute-force)", "ML (Surrogate)"]
    search_times = [classical["time_seconds"], ml["time_seconds"]]

    # --- Extract best runtime + memory ---
    best_classical = classical["best_actual"]
    best_ml = ml["winner_actual_verified"]

    runtime_labels = ["Best Classical Config", "ML Verified Config"]
    best_runtimes = [best_classical["runtime_s"], best_ml["runtime_s"]]
    best_memories = [best_classical["memory_mb"], best_ml["memory_mb"]]

    # --- Plot setup ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bar_width = 0.4
    indices = np.arange(2)

    # === (1) Search Time Comparison ===
    axes[0].bar(indices, search_times, color=["#E91E63", "#4CAF50"], width=bar_width)
    axes[0].set_title("Search Time Comparison", fontsize=14, fontweight="bold")
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(search_labels, rotation=10)
    axes[0].set_ylabel("Total Search Time (s)")
    axes[0].bar_label(axes[0].containers[0], fmt="%.2fs", padding=3)
    axes[0].text(
        0.5, max(search_times) * 0.9,
        f"≈ {speedup:.1f}× Faster",
        ha="center", fontsize=12, color="black", fontweight="bold"
    )

    # === (2) Runtime of Best Parameters ===
    axes[1].bar(indices, best_runtimes, color=["#2196F3", "#8BC34A"], width=bar_width)
    axes[1].set_title("Runtime of Best Parameters", fontsize=14, fontweight="bold")
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(runtime_labels, rotation=10)
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].bar_label(axes[1].containers[0], fmt="%.4fs", padding=3)

    # === (3) Memory Usage Comparison ===
    axes[2].bar(indices, best_memories, color=["#03A9F4", "#9C27B0"], width=bar_width)
    axes[2].set_title("Memory Usage of Best Parameters", fontsize=14, fontweight="bold")
    axes[2].set_xticks(indices)
    axes[2].set_xticklabels(runtime_labels, rotation=10)
    axes[2].set_ylabel("Memory (MB)")
    axes[2].bar_label(axes[2].containers[0], fmt="%.1f MB", padding=3)
    mem_ratio = best_classical["memory_mb"] / best_ml["memory_mb"]
    axes[2].text(
        0.5, max(best_memories) * 0.9,
        f"≈ {mem_ratio:.2f}× Smaller (ML)",
        ha="center", fontsize=12, color="black", fontweight="bold"
    )

    # === Global title ===
    plt.suptitle("Homomorphic Encryption Parameter Search: ML vs Classical", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # --- Save and show ---
    out_path = Path("artifacts/plot_search_comparison_full.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    # --- Print summary ---
    print("\n✅ Plot saved at:", out_path)
    print(f"⏱️ Classical Search Time: {search_times[0]:.2f}s")
    print(f"⚡ ML Search Time: {search_times[1]:.3f}s")
    print(f"🚀 Speedup: {speedup:.1f}× faster")
    print(f"🏁 Best Classical Runtime: {best_runtimes[0]:.5f}s")
    print(f"🤖 Best ML Verified Runtime: {best_runtimes[1]:.5f}s")
    print(f"💾 Memory Usage — Classical: {best_memories[0]:.1f} MB | ML: {best_memories[1]:.1f} MB "
          f"({mem_ratio:.2f}× smaller)")

if __name__ == "__main__":
    main()
