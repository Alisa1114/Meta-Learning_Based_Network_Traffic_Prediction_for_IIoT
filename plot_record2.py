import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
plt.rcParams.update({'font.size': 12})

plot_path = "exp_plot"
os.makedirs(plot_path, exist_ok=True)

maml_csv = "exp_plot/maml.csv"
baseline_csv = "exp_plot/flow2graph2.csv"

maml = pd.read_csv(maml_csv)
baseline = pd.read_csv(baseline_csv)

metric = ["RMSE", "MAPE", "MAE"]
# x_labels = ["SelfAtt\n+MAML", "Flow2graph"]
x_labels = ["SelfAtt\n+MAML", "F2G"]
datasets = [
    "F-cpu_usage",
    "F-memoory_free",
    "F-memoryallocatedbyproc",
    "F-in_traffic",
    "F-out_traffic",
]
models = "attn"

x = np.array([1, 5])
bar_width = 1.5
offset = bar_width / 2

# print(maml)

for d in datasets:
    select = d + "-M-"
    maml_results = maml[maml["Dataset"] == (select + models)]
    baseline_results = baseline[baseline["Dataset"] == (select + "flow2graph")]

    # print(maml_results)

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))  # horizontal
    # fig, axs = plt.subplots(3, 1, figsize=(5, 10)) # vertical

    for i, m in enumerate(metric):
        maml_y = maml_results[m].values
        baseline_y = baseline_results[m].values
        # print(maml_y, baseline_y)

        axs[i].bar(x[0], maml_y, width=bar_width, label="SelfAtt", color="blue")
        axs[i].bar(x[1], baseline_y, width=bar_width, label="Flow2graph", color="green")
        axs[i].set_xticklabels(x_labels)
        axs[i].set_xlim(x[0] - 1, x[1] + 1)
        axs[i].set_xticks(x)
        axs[i].set_ylabel(m)
        # axs[i].legend()
    
    plt.tight_layout()
    plt.savefig("{}/{}-vsflow2graph.png".format(plot_path, d))
    plt.close()
