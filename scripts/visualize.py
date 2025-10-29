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

