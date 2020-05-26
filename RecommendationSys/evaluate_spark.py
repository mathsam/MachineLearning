import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pylab as plt

baseline = pd.read_csv(r".\RecommendationSys\baseline_dim16.csv")
als = pd.read_csv(r".\RecommendationSys\als_full_dim16.csv")



baseline_fpr, baseline_tpr, thresholds = metrics.roc_curve(baseline['rating'] >=3, baseline['avg(rating)'])
baseline_auc = metrics.auc(baseline_fpr, baseline_tpr)

fpr, tpr, thresholds = metrics.roc_curve(als['rating'] >= 3, als['prediction'])
auc = metrics.auc(fpr, tpr)

plt.clf()
plt.plot(baseline_fpr, baseline_tpr, label='Baseline: AUC=%.2f' %baseline_auc)
plt.plot(fpr, tpr, label='ALS: AUC=%.2f' %auc)
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.legend(loc='lower right')
plt.show()