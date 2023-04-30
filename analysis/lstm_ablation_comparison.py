import ast
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

auroc = {}
acc = {}
with open('../generated_models/my_val_metric.txt', 'r') as ff:
    c = ff.read()
    my_val_metric = ast.literal_eval(c)
    metric_dict = my_val_metric[-1]
    tp = metric_dict['tp']
    tn = metric_dict['tn']
    fp = metric_dict['fp']
    fn = metric_dict['fn']
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
    acc['None'] = accuracy

    fpr, tpr, thr = metrics.roc_curve(metric_dict['labels'], metric_dict['preds'])
    auroc['None'] = metrics.auc(fpr, tpr)

with open('../generated_models/my_val_metric_noPooling.txt', 'r') as ff:
    c = ff.read()
    my_val_metric = ast.literal_eval(c)
    metric_dict = my_val_metric[-1]
    tp = metric_dict['tp']
    tn = metric_dict['tn']
    fp = metric_dict['fp']
    fn = metric_dict['fn']
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
    acc['NoPooling'] = accuracy

    fpr, tpr, thr = metrics.roc_curve(metric_dict['labels'], metric_dict['preds'])
    auroc['NoPooling'] = metrics.auc(fpr, tpr)

with open('../generated_models/my_val_metric_noTime.txt', 'r') as ff:
    c = ff.read()
    my_val_metric = ast.literal_eval(c)
    metric_dict = my_val_metric[-1]
    tp = metric_dict['tp']
    tn = metric_dict['tn']
    fp = metric_dict['fp']
    fn = metric_dict['fn']
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
    acc['NoTime'] = accuracy

    fpr, tpr, thr = metrics.roc_curve(metric_dict['labels'], metric_dict['preds'])
    auroc['NoTime'] = metrics.auc(fpr, tpr)

with open('../generated_models/my_val_metric_avgPool.txt', 'r') as ff:
    c = ff.read()
    my_val_metric = ast.literal_eval(c)
    metric_dict = my_val_metric[-1]
    tp = metric_dict['tp']
    tn = metric_dict['tn']
    fp = metric_dict['fp']
    fn = metric_dict['fn']
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
    acc['AvgPooling'] = accuracy

    fpr, tpr, thr = metrics.roc_curve(metric_dict['labels'], metric_dict['preds'])
    auroc['AvgPooling'] = metrics.auc(fpr, tpr)

with open('../generated_models/my_val_metric_noTime.txt', 'r') as ff:
    c = ff.read()
    my_val_metric = ast.literal_eval(c)
    metric_dict = my_val_metric[-1]
    tp = metric_dict['tp']
    tn = metric_dict['tn']
    fp = metric_dict['fp']
    fn = metric_dict['fn']
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
    acc['NoTime'] = accuracy

    fpr, tpr, thr = metrics.roc_curve(metric_dict['labels'], metric_dict['preds'])
    auroc['NoTime'] = metrics.auc(fpr, tpr)

ablations = list(acc.keys())
accuracies = list(acc.values())
aurocs = list(auroc.values())

colors = plt.cm.viridis(np.linspace(0, 1, len(ablations)))
plt.bar(ablations, accuracies, color=colors)

plt.title("Ablation Accuracies")
plt.xlabel("Ablations")
plt.ylabel("Accuracy")

plt.ylim(0, 1)
plt.yticks([i/10 for i in range(0, 11)], ['{:.0f}%'.format(i * 100) for i in [i/10 for i in range(0, 11)]])
plt.xticks(fontsize=8)
plt.savefig("ablation_accuracies.png", dpi=300, bbox_inches="tight")
plt.show()


plt.bar(ablations, aurocs, color=colors)

plt.title("Ablation AUROCs")
plt.xlabel("Ablations")
plt.ylabel("AUROC")

plt.ylim(0, 1)
plt.yticks([i/10 for i in range(0, 11)], [str(i) for i in [i/10 for i in range(0, 11)]])
plt.xticks(fontsize=8)
plt.savefig("ablation_aurocs.png", dpi=300, bbox_inches="tight")
plt.show()
