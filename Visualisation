features_only_df = features_df.drop(columns=['Patient_ID', 'Label'])

features_only_df.hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)

for ax, col in zip(plt.gcf().axes, features_only_df.columns):
    ax.set_xlabel(col)  

plt.tight_layout()
plt.show()

#  6 features at a time
feature_groups = [features_only_df.columns[i:i + 6] for i in range(0, len(features_only_df.columns), 6)]

for group in feature_groups:
    sns.pairplot(features_df, vars=group, hue="Label", diag_kind="kde", plot_kws={'s': 10})
    plt.suptitle(f"Pairplot for Features: {', '.join(group)}", y=1.02)
    plt.show()

# Spearman's correlation matrix 
spearman_corr = features_only_df.corr(method='spearman')
print("Spearman's Correlation Matrix:")
print(spearman_corr)

# Heatmap of Spearman's correlation 
plt.figure(figsize=(14, 12))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
plt.title("Spearman's Correlation Heatmap")
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)  
plt.tight_layout()
plt.show()
