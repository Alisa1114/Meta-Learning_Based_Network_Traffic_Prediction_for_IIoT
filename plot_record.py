import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
plt.rcParams.update({'font.size': 12})

plot_path = "exp_plot"
os.makedirs(plot_path, exist_ok=True)

pred_type = ""
maml_csv = "exp_plot/maml{}.csv".format(pred_type)
baseline_csv = "exp_plot/baseline{}.csv".format(pred_type)

maml = pd.read_csv(maml_csv)
baseline = pd.read_csv(baseline_csv)

metric = ["RMSE", "MAPE", "MAE"]
x_labels = ["CNN", "Self-Att", "LSTM"]
datasets = [
    "F-cpu_usage",
    "F-memoory_free",
    "F-memoryallocatedbyproc",
    "F-in_traffic",
    "F-out_traffic",
    # "Guangzhou",
    # "Seattle",
]
if pred_type == "":
    datasets.append("Guangzhou")
    datasets.append("Seattle")
    
models = ["cnn", "attn", "lstm"]

x = np.array([1, 3, 5])
bar_width = 0.5
offset = bar_width / 2

for d in datasets:
    maml_results = maml[maml["Dataset"].str.contains(d, na=False)]
    baseline_results = baseline[baseline["Dataset"].str.contains(d, na=False)]

    fig, axs = plt.subplots(1, 3, figsize=(8.1, 3))  # horizontal
    # fig, axs = plt.subplots(3, 1, figsize=(5, 10)) # vertical

    for i, m in enumerate(metric):
        maml_y = maml_results[m].values
        baseline_y = baseline_results[m].values

        axs[i].bar(x - offset, maml_y, width=bar_width, label="MAML", color="blue")
        axs[i].bar(
            x + offset, baseline_y, width=bar_width, label="Baseline", color="red"
        )
        axs[i].set_xticklabels(x_labels)
        axs[i].set_xticks(x)
        axs[i].set_ylabel(m)
        # axs[i].legend()

    
    plt.legend(loc="upper center", bbox_to_anchor=(0.7,1.5))
    plt.tight_layout()

    if d != "Guangzhou" and d != "Seattle":
        if pred_type == "":
            plt.savefig("{}/{}-M.png".format(plot_path, d), bbox_inches="tight")
        else:
            plt.savefig("{}/{}-V.png".format(plot_path, d), bbox_inches="tight")
    else:
        plt.savefig("{}/{}.png".format(plot_path, d), bbox_inches="tight")
    plt.close()
