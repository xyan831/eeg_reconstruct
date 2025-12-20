import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# test code
visual_path = "result/visual"
log_path = "result/log"
pdfname = "training_comparison_graphs.pdf"
pdf_path = os.path.join(visual_path, pdfname)

def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

# ----------------------------------------------
# Load ALL log files into dictionary
# ----------------------------------------------
def load_log(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metrics = lines[0].strip().split(",")
    metric_vals = {m: [] for m in metrics}

    for line in lines[1:]:
        vals = [to_num(x) for x in line.strip().split(",")]
        for m, v in zip(metrics, vals):
            metric_vals[m].append(v)

    return metrics, metric_vals

# ----------------------------------------------
# Group logs by dataset + task
# dataset: our / nicu
# task: recon / class
# ----------------------------------------------
groups = {}   # groups[(dataset, task)] = list of (name, metrics, values)

txt_files = [f for f in os.listdir(log_path) if f.endswith(".txt")]
for log_name in txt_files:
    full_path = os.path.join(log_path, log_name)

    dataset = "our" if "our" in log_name else "nicu"
    task = "recon" if "recon" in log_name else "class"

    metrics, metric_val = load_log(full_path)
    groups.setdefault((dataset, task), []).append(
        (log_name.replace(".txt", ""), metrics, metric_val)
    )

# ----------------------------------------------
# Create comparison graphs for each group
# ----------------------------------------------
pdf = PdfPages(pdf_path)

for (dataset, task), logs in groups.items():

    # Determine which metrics to plot (skip epoch)
    metric_list = logs[0][1][1:]  # all metrics except first (epoch)

    n_plots = len(metric_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for i, metric in enumerate(metric_list):
        ax = axes[i]

        for log_name, metrics, metric_val in logs:
            ax.plot(
                metric_val[metrics[0]],
                metric_val[metric],
                marker="o",
                label=log_name
            )

        ax.set_title(f"{dataset.upper()} - {task.upper()} : {metric} vs {metrics[0]}")
        ax.set_xlabel(metrics[0])
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

pdf.close()
print(f"Saved comparison graphs to {pdfname}")

