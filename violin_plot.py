from cProfile import label
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

    
data = [final_sols[alg] for alg in algoritmos]

df = pd.DataFrame(data.T, columns=algoritmos)

ax = sns.violinplot(data=df, palette=sns.color_palette(), cut=0, inner=None)
for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path( plt.Rectangle((x0, y0), width/2, height, transform=ax.transData))
num_items = len(ax.collections)
sns.stripplot(data=df, palette=sns.color_palette(), alpha=0.4, size=7)
for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.15)
sns.boxplot(data=df, width=0.25, showfliers=False, showmeans=True, meanprops=dict(marker='o', markerfacecolor='darkorange', markersize=10, zorder=3),
boxprops=dict(facecolor=(0,0,0,0), linewidth=3, zorder=3), whiskerprops=dict(linewidth=3), capprops=dict(linewidth=3), medianprops=dict(linewidth=3))
plt.legend(frameon=False, fontsize=15, loc='lower left')
# plt.title('', fontsize=28)
plt.title('Sphere + $N(0, 1)$')
plt.xlabel('Algoritmos')
plt.ylabel('Valor de função')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# sns.despine()
plt.savefig('objvls.pdf')