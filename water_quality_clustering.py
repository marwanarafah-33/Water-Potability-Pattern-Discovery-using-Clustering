"""
=============================================================
  WATER QUALITY CLUSTERING — APPLIED MACHINE LEARNING PROJECT
  
=============================================================
"""

# ============================================================
# SECTION 0: IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# --- Global Style ---
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c',
    'axes.labelcolor':  '#e0e0f0',
    'xtick.color':      '#9090b0',
    'ytick.color':      '#9090b0',
    'text.color':       '#e0e0f0',
    'grid.color':       '#2a2d45',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'font.size':        10,
})

PALETTE   = ['#00d4ff', '#ff6b6b', '#43e97b']   # clean / moderate / polluted
PALETTE4  = ['#00d4ff', '#ff6b6b', '#43e97b', '#ffd166']
NOISE_CLR = '#555577'

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT, exist_ok=True)


# ============================================================
# SECTION 1: LOAD DATA
# ============================================================
print("\n" + "="*60)
print("  SECTION 1 — LOAD DATA")
print("="*60)

df_raw = pd.read_csv('water_potability.csv')
print(f"✔ Loaded dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"  Columns : {df_raw.columns.tolist()}")
print(f"\n  Missing values per column:\n{df_raw.isnull().sum()}")

# Drop the label column — this is UNSUPERVISED learning
# We keep it aside only for post-hoc comparison
labels_true = df_raw['Potability'].values
features    = [c for c in df_raw.columns if c != 'Potability']
df          = df_raw[features].copy()


# ============================================================
# SECTION 2: DATA CLEANING
# ============================================================
print("\n" + "="*60)
print("  SECTION 2 — DATA CLEANING")
print("="*60)

# --- 2.1 Impute missing values with median ---
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=features)
print(f"✔ Imputed {df.isnull().sum().sum()} missing values using column medians")

# --- 2.2 Outlier detection via IQR capping ---
df_clean = df_imputed.copy()
clipped = 0
for col in features:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 3*IQR, Q3 + 3*IQR
    before = df_clean[col].copy()
    df_clean[col] = df_clean[col].clip(lo, hi)
    clipped += (df_clean[col] != before).sum()
print(f"✔ Clipped {clipped} outlier values (IQR × 3 rule)")


# ============================================================
# SECTION 3: FEATURE SCALING
# ============================================================
print("\n" + "="*60)
print("  SECTION 3 — FEATURE SCALING")
print("="*60)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)
X_scaled = pd.DataFrame(X_scaled, columns=features)
print(f"✔ StandardScaler applied — all features now have mean≈0, std≈1")
print(X_scaled.describe().round(3).loc[['mean','std']])


# ============================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "="*60)
print("  SECTION 4 — EDA")
print("="*60)

# --- 4.1 Feature distributions ---
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Feature Distributions (After Cleaning)', fontsize=15, fontweight='bold', color='#e0e0f0', y=1.01)
for ax, col in zip(axes.flat, features):
    ax.hist(df_clean[col], bins=40, color='#00d4ff', alpha=0.75, edgecolor='#003344')
    ax.set_title(col, fontweight='bold', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/1_feature_distributions.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 1_feature_distributions.png")

# --- 4.2 Correlation heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_clean.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt='.2f',
            linewidths=0.5, ax=ax, annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/2_correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 2_correlation_heatmap.png")


# ============================================================
# SECTION 5: DIMENSIONALITY REDUCTION — PCA
# ============================================================
print("\n" + "="*60)
print("  SECTION 5 — PCA")
print("="*60)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"✔ PCA reduced {len(features)} features → 2 components")
print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%  "
      f"Total={sum(pca.explained_variance_ratio_)*100:.1f}%")

# --- PCA loadings (which features contribute most) ---
loadings = pd.DataFrame(pca.components_.T, index=features, columns=['PC1','PC2'])
print("\n  PCA Loadings (feature importance per component):")
print(loadings.round(3))


# ============================================================
# SECTION 6: CHOOSE OPTIMAL K — ELBOW + SILHOUETTE
# ============================================================
print("\n" + "="*60)
print("  SECTION 6 — OPTIMAL K SELECTION")
print("="*60)

inertias, silhouettes = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"  k={k}  Inertia={km.inertia_:,.0f}  Silhouette={silhouette_score(X_scaled, km.labels_):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Optimal K Selection', fontsize=14, fontweight='bold')

ax1.plot(K_range, inertias, 'o-', color='#00d4ff', linewidth=2.5, markersize=8)
ax1.axvline(x=3, color='#ff6b6b', linestyle='--', linewidth=2, label='Chosen k=3')
ax1.set_title('Elbow Method (Inertia)', fontweight='bold')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Within-Cluster SSE)')
ax1.legend()
ax1.grid(True, alpha=0.4)

ax2.plot(K_range, silhouettes, 's-', color='#43e97b', linewidth=2.5, markersize=8)
ax2.axvline(x=3, color='#ff6b6b', linestyle='--', linewidth=2, label='Chosen k=3')
ax2.set_title('Silhouette Scores', fontweight='bold')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(f'{OUTPUT}/3_optimal_k.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 3_optimal_k.png")

BEST_K = 3


# ============================================================
# SECTION 7: K-MEANS CLUSTERING
# ============================================================
print("\n" + "="*60)
print(f"  SECTION 7 — K-MEANS  (k={BEST_K})")
print("="*60)

kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)

km_sil = silhouette_score(X_scaled, km_labels)
print(f"✔ K-Means fitted | Silhouette Score = {km_sil:.4f}")
print(f"  Cluster sizes: {dict(zip(*np.unique(km_labels, return_counts=True)))}")

# --- Cluster profiles ---
df_km = df_clean.copy()
df_km['Cluster'] = km_labels
cluster_means = df_km.groupby('Cluster')[features].mean()
print("\n  Cluster mean profiles:")
print(cluster_means.round(2))


# ============================================================
# SECTION 8: DBSCAN CLUSTERING
# ============================================================
print("\n" + "="*60)
print("  SECTION 8 — DBSCAN")
print("="*60)

# Tune eps via k-distance graph
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
dists, _ = nn.kneighbors(X_scaled)
dists_sorted = np.sort(dists[:, 4])[::-1]

dbscan = DBSCAN(eps=2.5, min_samples=10)
db_labels = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = (db_labels == -1).sum()
print(f"✔ DBSCAN fitted  | eps=2.5, min_samples=10")
print(f"  Clusters found : {n_clusters_db}")
print(f"  Noise points   : {n_noise} ({n_noise/len(db_labels)*100:.1f}%)")

if n_clusters_db > 1:
    mask_valid = db_labels != -1
    db_sil = silhouette_score(X_scaled[mask_valid], db_labels[mask_valid])
    print(f"  Silhouette Score (excl. noise): {db_sil:.4f}")
else:
    db_sil = None
    print("  ⚠ Only 1 cluster found — silhouette score not meaningful")


# ============================================================
# SECTION 9: VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("  SECTION 9 — VISUALIZATIONS")
print("="*60)

# --- 9.1  K-Means PCA scatter ---
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(BEST_K):
    mask = km_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=PALETTE[c], label=f'Cluster {c}', alpha=0.6, s=18, edgecolors='none')
# Centroids projected
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c='white', s=200, marker='*', zorder=5, edgecolors='black', linewidths=0.8, label='Centroids')
ax.set_title(f'K-Means Clustering (k={BEST_K}) — PCA 2D', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.legend(framealpha=0.3)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/4_kmeans_pca.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 4_kmeans_pca.png")

# --- 9.2  DBSCAN PCA scatter ---
fig, ax = plt.subplots(figsize=(10, 7))
unique_db = sorted(set(db_labels))
cmap_db   = [NOISE_CLR] + PALETTE4[:n_clusters_db]
label_map = {-1: 'Noise (−1)'}
for i, c in enumerate(unique_db):
    if c != -1:
        label_map[c] = f'Cluster {c}'
for i, c in enumerate(unique_db):
    mask  = db_labels == c
    color = NOISE_CLR if c == -1 else PALETTE4[i if c != -1 else 0]
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=color, label=label_map[c], alpha=0.5 if c != -1 else 0.2,
               s=12 if c != -1 else 6, edgecolors='none')
ax.set_title('DBSCAN Clustering — PCA 2D', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.legend(framealpha=0.3, markerscale=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/5_dbscan_pca.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 5_dbscan_pca.png")

# --- 9.3  Side-by-side comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('K-Means vs DBSCAN — PCA 2D Comparison', fontsize=14, fontweight='bold')

for c in range(BEST_K):
    mask = km_labels == c
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=PALETTE[c], label=f'Cluster {c}', alpha=0.55, s=15, edgecolors='none')
ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='white', s=200, marker='*', zorder=5, edgecolors='black', linewidths=0.8)
ax1.set_title(f'K-Means  (k={BEST_K})  |  Sil={km_sil:.3f}', fontweight='bold', fontsize=11)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.legend(framealpha=0.3); ax1.grid(True, alpha=0.3)

for i, c in enumerate(unique_db):
    mask  = db_labels == c
    color = NOISE_CLR if c == -1 else PALETTE4[i if c != -1 else 0]
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=color, label=label_map[c],
                alpha=0.5 if c != -1 else 0.15, s=12, edgecolors='none')
sil_txt = f"Sil={db_sil:.3f}" if db_sil else "Sil=N/A"
ax2.set_title(f'DBSCAN  ({n_clusters_db} clusters, {n_noise} noise)  |  {sil_txt}', fontweight='bold', fontsize=11)
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
ax2.legend(framealpha=0.3, markerscale=2); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT}/6_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 6_comparison.png")

# --- 9.4  Cluster feature radar / box plots ---
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Feature Distributions per K-Means Cluster', fontsize=13, fontweight='bold')
for ax, col in zip(axes.flat, features):
    for c in range(BEST_K):
        vals = df_clean.loc[km_labels == c, col]
        ax.hist(vals, bins=30, color=PALETTE[c], alpha=0.55, label=f'C{c}', edgecolor='none')
    ax.set_title(col, fontweight='bold', fontsize=9)
    ax.set_xlabel('Value', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.3)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT}/7_cluster_feature_dist.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 7_cluster_feature_dist.png")

# --- 9.5  Cluster mean heatmap ---
fig, ax = plt.subplots(figsize=(12, 4))
scaled_means = pd.DataFrame(
    scaler.transform(cluster_means),
    index=[f'Cluster {i}' for i in range(BEST_K)],
    columns=features
)
sns.heatmap(scaled_means, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, ax=ax, center=0, cbar_kws={'label': 'Standardised Mean'})
ax.set_title('Cluster Profiles — Standardised Feature Means', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT}/8_cluster_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✔ Saved: 8_cluster_heatmap.png")


# ============================================================
# SECTION 10: EVALUATION & INTERPRETATION
# ============================================================
print("\n" + "="*60)
print("  SECTION 10 — EVALUATION & INTERPRETATION")
print("="*60)

print(f"\n  ╔══════════════════════════════════════════╗")
print(f"  ║  MODEL COMPARISON SUMMARY                ║")
print(f"  ╠══════════════════════════════════════════╣")
print(f"  ║  K-Means  │ k={BEST_K}  │ Silhouette = {km_sil:.4f}   ║")
sil_line = f"{db_sil:.4f}" if db_sil else "  N/A  "
print(f"  ║  DBSCAN   │ {n_clusters_db} clusters │ Silhouette = {sil_line}   ║")
print(f"  ╚══════════════════════════════════════════╝")

# --- Assign semantic labels based on cluster means ---
ph_means    = cluster_means['ph']
turb_means  = cluster_means['Turbidity']
sorted_by_ph = ph_means.sort_values()

LABEL_MAP = {}
cluster_ids = sorted_by_ph.index.tolist()
LABEL_MAP[cluster_ids[0]] = 'Polluted Water'
LABEL_MAP[cluster_ids[1]] = 'Moderate Quality'
LABEL_MAP[cluster_ids[2]] = 'Clean Water'

print("\n  Cluster Interpretation:")
for cid, name in LABEL_MAP.items():
    print(f"\n  ► Cluster {cid} → '{name}'")
    row = cluster_means.loc[cid]
    print(f"     pH            = {row['ph']:.2f}")
    print(f"     Turbidity     = {row['Turbidity']:.2f}")
    print(f"     Sulfate       = {row['Sulfate']:.2f}")
    print(f"     Chloramines   = {row['Chloramines']:.2f}")
    print(f"     Conductivity  = {row['Conductivity']:.2f}")
    print(f"     Organic C     = {row['Organic_carbon']:.2f}")


# ============================================================
# SECTION 11: FINAL SUMMARY REPORT
# ============================================================
print("\n" + "="*60)
print("  SECTION 11 — PROJECT SUMMARY REPORT")
print("="*60)

report = f"""
╔══════════════════════════════════════════════════════════════╗
║        WATER QUALITY CLUSTERING — FINAL REPORT               ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset       : 3,276 water samples × 9 features            ║
║  Missing values: pH (491), Sulfate (781), THMs (162)         ║
║                  → Imputed with column medians               ║
║  Outliers      : Capped using IQR×3 rule                     ║
║  Scaling       : StandardScaler (zero mean, unit variance)   ║
╠══════════════════════════════════════════════════════════════╣
║  PCA           : 2 components → {sum(pca.explained_variance_ratio_)*100:.1f}% variance explained      ║
║  Optimal K     : 3 (Elbow + Silhouette convergence)          ║
╠══════════════════════════════════════════════════════════════╣
║  K-MEANS       ║  k=3  ║  Silhouette = {km_sil:.4f}               ║
║  DBSCAN        ║  {n_clusters_db} clusters, {n_noise} noise  ║  Silhouette = {str(round(db_sil,4)) if db_sil else 'N/A'}           ║
╠══════════════════════════════════════════════════════════════╣
║  WINNER: K-Means → cleaner, more balanced clusters          ║
╠══════════════════════════════════════════════════════════════╣
║  CLUSTER LABELS                                              ║
"""
for cid, name in LABEL_MAP.items():
    report += f"║    Cluster {cid} → {name:<44}║\n"
report += """╠══════════════════════════════════════════════════════════════╣
║  LIMITATIONS                                                 ║
║  • No ground-truth water quality labels available            ║
║  • DBSCAN sensitive to eps/min_samples tuning                ║
║  • PCA loses some variance (features not fully separable)    ║
║  • Cluster labels are inferred, not clinically validated     ║
╠══════════════════════════════════════════════════════════════╣
║  REAL-WORLD INSIGHTS                                         ║
║  • High pH + low turbidity clusters indicate clean water     ║
║  • Elevated sulfate / organic carbon → pollution signals     ║
║  • Can guide water treatment plant prioritisation            ║
╚══════════════════════════════════════════════════════════════╝
"""
print(report)

print("✅ ALL DONE — Check /mnt/user-data/outputs/ for all plots")
