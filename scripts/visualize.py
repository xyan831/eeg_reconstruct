import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# compare pred and true for multi channel output
def plot_pred_stack(y_pred, y_test, X_test, ch_list, sample=0):
    for i in ch_list:
        plt.figure()
        plt.plot(X_test[sample, i-1, :], label="Input")
        plt.plot(y_test[sample, i-1, :], label="True")
        plt.plot(y_pred[sample, i-1, :], label="Pred")
        plt.legend()
        plt.title(f"True vs Predicted output channel{i} [sample {sample}]")
        plt.show()

# side by side compare graph
def plot_pred_side(y_pred, y_test, X_test, ch_list, sample=0):
    for i in ch_list:
        print(f"True vs Predicted output channel{i} [sample {sample}]")
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
        axes[0].plot(y_pred[sample, i-1, :])
        axes[0].set_title('Pred')
        axes[1].plot(y_test[sample, i-1, :])
        axes[1].set_title('True')
        axes[2].plot(X_test[sample, i-1, :])
        axes[2].set_title('Input')
        plt.tight_layout()
        plt.show()

# compare pred and true save to pdf
def save_pred_stack(pdfname, y_pred, y_test, X_test, ch_list, sample=0):
    with PdfPages(pdfname) as pdf:
        for i in ch_list:
            plt.figure()
            plt.plot(X_test[sample, i-1, :], label="Input")
            plt.plot(y_test[sample, i-1, :], label="True")
            plt.plot(y_pred[sample, i-1, :], label="Pred")
            plt.legend()
            plt.title(f"True vs Predicted output channel{i}")
            plt.savefig()
            plf.close()

# side by side compare graph save to pdf
def save_pred_side(pdfname, y_pred, y_test, X_test, ch_list, sample=0):
    with PdfPages(pdfname) as pdf:
        for i in ch_list:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
            axes[0].plot(y_pred[sample, i-1, :])
            axes[0].set_title('Pred')
            axes[1].plot(y_test[sample, i-1, :])
            axes[1].set_title('True')
            axes[2].plot(X_test[sample, i-1, :])
            axes[2].set_title('Input')
            plt.tight_layout()
            plt.suptitle(f"True vs Predicted output channel{i}", y=1)
            pdf.savefig()
            plt.close()

# graph model train log loss and accuracy by epoch
def log_graph(visual_path, log_path, log_name):
    pdf_name = f"train_{log_name.removesuffix('.txt')}.pdf"
    pdfname = os.path.join(visual_path, pdf_name)
    log_file = os.path.join(log_path, log_name)
    
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    def to_num(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
    metrics = lines[0].strip().split(",")
    metric_val = [[] for _ in metrics]
    for line in lines[1:]:
        strings = line.strip().split(",")
        values = [to_num(x) for x in strings]
        for i in range(len(values)):
            metric_val[i].append(values[i])
    
    met_num = 3 if "class" in log_name else 2
    n_plots = met_num - 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 6*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        y_index = i + 1  # metrics[1], metrics[2], ...
        ax.plot(metric_val[0], metric_val[y_index], marker="o", label=metrics[y_index])
        ax.set_xlabel(metrics[0])
        ax.set_ylabel(metrics[y_index])
        ax.set_title(f"{metrics[y_index]} vs {metrics[0]}")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig(pdfname)
    plt.close()
    print(f"Saved graph to {pdf_name}")





