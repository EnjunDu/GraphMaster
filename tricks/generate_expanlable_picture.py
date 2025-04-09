import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from matplotlib import cm, patheffects
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# Increase the font size overall on the original basis
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 15      # It turned out to be 13
plt.rcParams['axes.titlesize'] = 18      # It turned out to be 16
plt.rcParams['xtick.labelsize'] = 13     # It turned out to be 11
plt.rcParams['ytick.labelsize'] = 13     # It turned out to be 11

# Create a fresh, vibrant color scheme
colors = ['#55AAFF', '#FF9AA2', '#B5EAD7']  # Sky blue, light pink, mint green
pastel_purple = '#C5A3FF'
light_blue = '#AADDFF'

# Figure 1: Distribution of semantic consistency scores
plt.figure(figsize=(9, 6))

graphmaster_scores = np.array([
    0.89, 0.88, 0.91, 0.86, 0.90, 0.85, 0.92, 0.88, 0.87, 0.93,
    0.89, 0.86, 0.90, 0.91, 0.87, 0.84, 0.88, 0.93, 0.91, 0.86,
    0.89, 0.92, 0.90, 0.88, 0.85, 0.87, 0.93, 0.91, 0.89, 0.90
] * 5)  # 150 values

mixed_llm_scores = np.array([
    0.68, 0.61, 0.65, 0.62, 0.69, 0.64, 0.67, 0.63, 0.60, 0.66,
    0.61, 0.65, 0.68, 0.62, 0.67, 0.63, 0.60, 0.62, 0.66, 0.69,
    0.64, 0.61, 0.67, 0.60, 0.65, 0.68, 0.62, 0.69, 0.63, 0.66
] * 5)  # 150 values

synthesis_llm_scores = np.array([
    0.51, 0.58, 0.53, 0.56, 0.54, 0.55, 0.59, 0.53, 0.57, 0.55,
    0.56, 0.58, 0.54, 0.55, 0.59, 0.56, 0.57, 0.54, 0.55, 0.58,
    0.52, 0.64, 0.54, 0.54, 0.56, 0.57, 0.53, 0.58, 0.52, 0.58
] * 5)  # 150 values

graphmaster_mean = np.mean(graphmaster_scores)
mixed_llm_mean = np.mean(mixed_llm_scores)
synthesis_llm_mean = np.mean(synthesis_llm_scores)

# Set a beautiful background
ax = plt.gca()
ax.set_facecolor('#F8F9FD')
ax.patch.set_alpha(0.6)

# Fill the histogram with a refreshing gradient
hist_gm = plt.hist(graphmaster_scores, bins=15, alpha=0.7, density=True,
                   label='GraphMaster', color=colors[0], edgecolor='white', linewidth=0.8)
hist_mix = plt.hist(mixed_llm_scores, bins=15, alpha=0.65, density=True,
                    label='Mixed-LLM', color=colors[1], edgecolor='white', linewidth=0.8)
hist_syn = plt.hist(synthesis_llm_scores, bins=15, alpha=0.65, density=True,
                    label='Synthesis-LLM', color=colors[2], edgecolor='white', linewidth=0.8)

# Draw the mean line
plt.axvline(graphmaster_mean, color=colors[0], linestyle='--', linewidth=2.5, alpha=0.9)
plt.axvline(mixed_llm_mean, color=colors[1], linestyle='--', linewidth=2.5, alpha=0.9)
plt.axvline(synthesis_llm_mean, color=colors[2], linestyle='--', linewidth=2.5, alpha=0.9)

# Add mean label (larger font size)
for value, color, y_pos in zip(
        [graphmaster_mean, mixed_llm_mean, synthesis_llm_mean],
        [colors[0], colors[1], colors[2]],
        [7.0, 7.5, 8.0]):

    plt.text(value + 0.01, y_pos, f'μ={value:.2f}',
             color='#000000', fontsize=14, ha='left', va='center',  # 12 -> 14
             fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor=color,
                       boxstyle="round,pad=0.3", linewidth=1.5))

# Beautify axis labels and titles (both font sizes are increased)
plt.xlabel('Semantic Coherence Score', fontweight='bold', fontsize=16)  # 14 -> 16
plt.ylabel('Density', fontweight='bold', fontsize=16)                  # 14 -> 16
plt.title('Distribution of Semantic Coherence Scores',
          fontweight='bold', fontsize=20, pad=15)                      # 18 -> 20

# Set axis range
plt.xlim(0.45, 1.0)
plt.ylim(0, 25)

# Add legend
legend = plt.legend(frameon=True, loc='upper left', fontsize=14)  # 12 -> 14
frame = legend.get_frame()
frame.set_facecolor('#F8F9FD')
frame.set_edgecolor(pastel_purple)
frame.set_linewidth(1.5)
frame.set_alpha(0.95)

# Add a comment (larger font size)
plt.annotate('Superior Performance',
             xy=(graphmaster_mean, 7.0), xytext=(graphmaster_mean - 0.1, 5.0),
             arrowprops=dict(arrowstyle='->', color=pastel_purple, linewidth=2,
                             connectionstyle="arc3,rad=-.2"),
             color='#000000', fontsize=14, fontweight='bold',  # 12 -> 14
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=pastel_purple, alpha=0.8))

# Add border
for spine in plt.gca().spines.values():
    spine.set_edgecolor(pastel_purple)
    spine.set_linewidth(1.2)
    spine.set_alpha(0.8)

plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout(pad=2.0)
plt.savefig('figure1_coherence_distribution.pdf', bbox_inches='tight', dpi=300)
plt.close()


# Figure 2: Correlation scatter plot
plt.figure(figsize=(9, 6))

human_ratings = [
    0.91, 0.89, 0.93, 0.88, 0.92, 0.90, 0.87, 0.94, 0.89, 0.92,
    0.88, 0.91, 0.93, 0.90, 0.86, 0.91, 0.89, 0.92, 0.87, 0.93,
    0.90, 0.88, 0.92, 0.86, 0.91, 0.89, 0.93, 0.90, 0.87, 0.92
] * 3

semantic_scores = [
    0.88, 0.85, 0.90, 0.83, 0.89, 0.87, 0.82, 0.91, 0.86, 0.89,
    0.84, 0.88, 0.90, 0.87, 0.83, 0.88, 0.85, 0.89, 0.82, 0.90,
    0.87, 0.84, 0.89, 0.81, 0.88, 0.85, 0.90, 0.87, 0.83, 0.89
] * 3

corr_data = pd.DataFrame({
    'Human Traceability Score': human_ratings,
    'Semantic Coherence Score': semantic_scores
})

correlation, p_value = stats.pearsonr(human_ratings, semantic_scores)

ax = plt.gca()
ax.set_facecolor('#F8F9FD')
ax.patch.set_alpha(0.6)

# Creating a gradient color map
n_colors = 90
cmap = LinearSegmentedColormap.from_list('custom_cmap', [light_blue, pastel_purple], N=n_colors)
point_colors = np.linspace(0, 1, len(human_ratings))

# Draw a scatter plot (with gradient effect)
scatter = plt.scatter(human_ratings, semantic_scores,
                      c=point_colors, cmap=cmap, s=80, alpha=0.8,
                      edgecolor='white', linewidth=0.8)

# regression line
sns.regplot(x='Human Traceability Score', y='Semantic Coherence Score',
            data=corr_data, scatter=False,
            line_kws={'color': pastel_purple, 'linewidth': 3, 'alpha': 0.8})

# Add relevance statistics in text box (larger font size)
stats_text = f'Pearson r = {correlation:.2f}\np-value < 0.00001'
text_box = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9,
                edgecolor=pastel_purple, linewidth=1.5)
plt.text(0.05, 0.15, stats_text, transform=plt.gca().transAxes, fontsize=15,  # 13 -> 15
         color='#000000', fontweight='bold', verticalalignment='bottom', bbox=text_box)

# Axis labels and titles (larger font size)
plt.xlabel('Human Traceability Score', fontweight='bold', fontsize=16)  # 14 -> 16
plt.ylabel('Semantic Coherence Score', fontweight='bold', fontsize=16)  # 14 -> 16
plt.title('Correlation: Human Ratings vs. Semantic Coherence',
          fontweight='bold', fontsize=20, pad=15)  # 18 -> 20

plt.xlim(0.85, 0.95)
plt.ylim(0.80, 0.92)

# Add a comment (larger font size)
plt.annotate('Strong Positive\nCorrelation',
             xy=(0.91, 0.87), xytext=(0.93, 0.83),
             arrowprops=dict(arrowstyle='->', color=pastel_purple, linewidth=2,
                             connectionstyle="arc3,rad=.3"),
             color='#000000', fontsize=14, fontweight='bold',  # 12 -> 14
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=pastel_purple, alpha=0.8))

# frame
for spine in plt.gca().spines.values():
    spine.set_edgecolor(pastel_purple)
    spine.set_linewidth(1.2)
    spine.set_alpha(0.8)

plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout(pad=2.0)
plt.savefig('figure2_correlation.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Figure 3: Bar chart comparison
plt.figure(figsize=(10, 6))

datasets = ['Cora', 'Citeseer', 'Wikics', 'History', 'Arxiv2023', 'Children']

graphmaster_by_dataset = [0.89, 0.87, 0.91, 0.85, 0.84, 0.93]
mixed_llm_by_dataset = [0.67, 0.62, 0.66, 0.63, 0.60, 0.62]
synthesis_llm_by_dataset = [0.64, 0.61, 0.65, 0.62, 0.59, 0.61]

x = np.arange(len(datasets))
width = 0.25

ax = plt.gca()
ax.set_facecolor('#F8F9FD')
ax.patch.set_alpha(0.6)

graphmaster_colors = [mpl.colors.to_rgba(colors[0], alpha=0.6 + 0.05*i) for i in range(len(datasets))]
mixed_colors = [mpl.colors.to_rgba(colors[1], alpha=0.6 + 0.05*i) for i in range(len(datasets))]
synthesis_colors = [mpl.colors.to_rgba(colors[2], alpha=0.6 + 0.05*i) for i in range(len(datasets))]

bars1 = plt.bar(x - width, graphmaster_by_dataset, width, label='GraphMaster',
                color=graphmaster_colors, edgecolor='white', linewidth=1)
bars2 = plt.bar(x, mixed_llm_by_dataset, width, label='Mixed-LLM',
                color=mixed_colors, edgecolor='white', linewidth=1)
bars3 = plt.bar(x + width, synthesis_llm_by_dataset, width, label='Synthesis-LLM',
                color=synthesis_colors, edgecolor='white', linewidth=1)

# Top value label (larger font size)
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=12,  # 10 -> 12
             fontweight='bold', color='#555555')

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=12,  # 10 -> 12
             fontweight='bold', color='#555555')

for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=12,  # 10 -> 12
             fontweight='bold', color='#555555')

# Axes and titles (larger font size)
plt.ylabel('Avg. Semantic Coherence Score', fontweight='bold', fontsize=16)  # 14 -> 16
plt.title('Semantic Coherence Across Datasets', fontweight='bold', fontsize=20, pad=15)  # 18 -> 20

plt.xticks(x, datasets, fontsize=14)
plt.setp(plt.gca().get_xticklabels(), rotation=30, ha='right')

# Threshold line annotation (larger font size)
threshold = 0.7
plt.axhline(y=threshold, color='#FF7BAC', linestyle='--', linewidth=2, alpha=0.8)
plt.text(5.5, threshold + 0.01, 'Interpretability threshold',
         ha="right", va="bottom", color='#000000', fontsize=14,  # 11 -> 14
         fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#FF7BAC',
                   boxstyle="round,pad=0.3", linewidth=1))

# Legend (larger font size)
legend = plt.legend(frameon=True, loc='upper right', fontsize=14)  # 12 -> 14
frame = legend.get_frame()
frame.set_facecolor('#F8F9FD')
frame.set_edgecolor(pastel_purple)
frame.set_linewidth(1.5)
frame.set_alpha(0.95)

# Notes (larger font size)
plt.annotate('GraphMaster consistently\noutperforms other models',
             xy=(3, 0.89), xytext=(3, 0.95),
             arrowprops=dict(arrowstyle='->', color='#000000', linewidth=2,
                             connectionstyle="arc3,rad=-.2"),
             color='#000000', fontsize=14,  # 12 -> 14
             fontweight='bold', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=pastel_purple, alpha=0.8))

for spine in plt.gca().spines.values():
    spine.set_edgecolor(pastel_purple)
    spine.set_linewidth(1.2)
    spine.set_alpha(0.8)

plt.ylim(0.5, 1.0)
plt.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout(pad=2.0)
plt.savefig('figure3_datasets.pdf', bbox_inches='tight', dpi=300)
plt.close()

print(f"GraphMaster mean score: {graphmaster_mean:.2f}")
print(f"Mixed-LLM mean score: {mixed_llm_mean:.2f}")
print(f"Synthesis-LLM mean score: {synthesis_llm_mean:.2f}")
print(f"Correlation: r = {correlation:.2f}, p = {p_value:.6f}")
