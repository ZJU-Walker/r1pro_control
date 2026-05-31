import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 数据定义
# -----------------------------
objects = ["redcube", "chicken", "pepper", "corn"]
algorithms = ["baseline", "cotrain", "ablation_no_keypoints"]

# SR (%) 数据
sr_id = {
    "baseline": [100, 86.67, 93.33, 80],
    "cotrain": [100, 100, 100, 93.33],
    "ablation_no_keypoints": [100, 86.67, 86.67, 86.67],
}
sr_ood = {
    "baseline": [26.67, 6.67, 13.33, 13.33],
    "cotrain": [93.33, 100, 93.33, 80],
    "ablation_no_keypoints": [46.67, 33.33, 26.67, 20],
}

# -----------------------------
# 图像样式设置
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")
colors = ["#4C72B0", "#55A868", "#C44E52"]
bar_width = 0.25
x = np.arange(len(objects))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# -----------------------------
# 绘制 ID 子图
# -----------------------------
for i, algo in enumerate(algorithms):
    axes[0].bar(x + i * bar_width, sr_id[algo], width=bar_width, color=colors[i], label=algo)

axes[0].set_title("PICK_PLACE (ID)", fontsize=13, fontweight="bold")
axes[0].set_xticks(x + bar_width)
axes[0].set_xticklabels(objects, fontsize=10)
axes[0].set_ylabel("SR (%)", fontsize=11)
axes[0].set_ylim(0, 110)
axes[0].grid(axis="y", linestyle="--", alpha=0.6)

# -----------------------------
# 绘制 OOD 子图
# -----------------------------
for i, algo in enumerate(algorithms):
    axes[1].bar(x + i * bar_width, sr_ood[algo], width=bar_width, color=colors[i], label=algo)

axes[1].set_title("PICK_PLACE (OOD)", fontsize=13, fontweight="bold")
axes[1].set_xticks(x + bar_width)
axes[1].set_xticklabels(objects, fontsize=10)
axes[1].set_ylim(0, 110)
axes[1].grid(axis="y", linestyle="--", alpha=0.6)

# -----------------------------
# 图例与布局
# -----------------------------
axes[1].legend(algorithms, fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

# -----------------------------
# 保存结果
# -----------------------------
plt.savefig("pick_place_comparison.pdf", bbox_inches="tight")
plt.savefig("pick_place_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
