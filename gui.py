"""
=============================================================
  WATER QUALITY CLUSTERING — GUI
  Applied Machine Learning
  Team: Marwan | Fares | Seif | Adham | Omar

  HOW TO RUN:
      streamlit run gui.py

  NOTE:
      water_quality_clustering.py and water_potability.csv
      make sure these files are in the same folder
=============================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Water Quality Clustering",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding: 2rem 3rem; }
    h1 { color: #00d4ff !important; }
    h2 { color: #00d4ff !important; border-bottom: 1px solid #2a2d45; padding-bottom: 6px; }
    h3 { color: #c0c0e0 !important; }
    p, li { color: #b0b0c8 !important; }

    .stTabs [data-baseweb="tab-list"] { background:#1a1d2e; border-radius:10px; padding:4px; }
    .stTabs [data-baseweb="tab"]      { color:#9090b0; border-radius:8px; padding:8px 20px; }
    .stTabs [aria-selected="true"]    { background:#00d4ff !important; color:#0f1117 !important; font-weight:bold; }

    [data-testid="stSidebar"]   { background:#1a1d2e; border-right:1px solid #2a2d45; }
    [data-testid="stSidebar"] * { color:#c0c0e0 !important; }

    [data-testid="stMetric"]      { background:#1a1d2e; border:1px solid #2a2d45; border-radius:10px; padding:1rem; }
    [data-testid="stMetricLabel"] { color:#9090b0 !important; }
    [data-testid="stMetricValue"] { color:#00d4ff !important; font-size:1.8rem !important; }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099bb);
        color:#0f1117 !important; font-weight:bold;
        border:none; border-radius:8px; padding:0.6rem 2rem; width:100%;
    }

    .card-clean    { background:#0a2a1a; border-left:4px solid #43e97b; padding:1rem 1.5rem; border-radius:8px; margin:8px 0; }
    .card-moderate { background:#2a2200; border-left:4px solid #ffd166; padding:1rem 1.5rem; border-radius:8px; margin:8px 0; }
    .card-polluted { background:#2a0000; border-left:4px solid #ff6b6b; padding:1rem 1.5rem; border-radius:8px; margin:8px 0; }
    .card-clean p, .card-moderate p, .card-polluted p { color:#e0e0e0 !important; margin:0; }

    .result-clean    { background:#0a2a1a; border:2px solid #43e97b; border-radius:12px; padding:2rem; text-align:center; }
    .result-moderate { background:#2a2200; border:2px solid #ffd166; border-radius:12px; padding:2rem; text-align:center; }
    .result-polluted { background:#2a0000; border:2px solid #ff6b6b; border-radius:12px; padding:2rem; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB DARK STYLE
# ============================================================
PALETTE  = ['#00d4ff', '#ff6b6b', '#43e97b']
PALETTE4 = ['#00d4ff', '#ff6b6b', '#43e97b', '#ffd166']
NOISE_CLR = '#555577'
BG   = '#1a1d2e'
TEXT = '#e0e0f0'
GRID = '#2a2d45'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': '#3a3d5c', 'axes.labelcolor': TEXT,
    'xtick.color': '#9090b0', 'ytick.color': '#9090b0',
    'text.color': TEXT, 'grid.color': GRID, 'grid.alpha': 0.5,
    'legend.facecolor': BG, 'legend.labelcolor': TEXT,
})


# ============================================================
# PIPELINE — same logic as water_quality_clustering.py
# ============================================================
@st.cache_data
def run_pipeline(file_source, k=3, eps=2.5, min_samples=10):
    # SECTION 1: LOAD
    if file_source is None:
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'water_potability.csv')
        df_raw = pd.read_csv(path)
    else:
        df_raw = pd.read_csv(file_source)

    labels_true = df_raw['Potability'].values if 'Potability' in df_raw.columns else None
    features    = [c for c in df_raw.columns if c != 'Potability']
    df          = df_raw[features].copy()

    # SECTION 2: CLEANING
    imputer    = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=features)

    df_clean = df_imputed.copy()
    for col in features:
        Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        df_clean[col] = df_clean[col].clip(Q1 - 3*IQR, Q3 + 3*IQR)

    # SECTION 3: SCALING
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    # SECTION 5: PCA
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled_df)

    # SECTION 6: OPTIMAL K
    inertias, silhouettes = [], []
    K_range = range(2, 11)
    for kk in K_range:
        km = KMeans(n_clusters=kk, random_state=42, n_init=10)
        km.fit(X_scaled_df)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled_df, km.labels_))

    # SECTION 7: K-MEANS
    kmeans    = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled_df)
    km_sil    = silhouette_score(X_scaled_df, km_labels)

    df_km = df_clean.copy()
    df_km['Cluster'] = km_labels
    cluster_means = df_km.groupby('Cluster')[features].mean()

    # SECTION 8: DBSCAN
    dbscan    = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_scaled_df)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise       = (db_labels == -1).sum()
    db_sil = None
    if n_clusters_db > 1:
        mask_valid = db_labels != -1
        db_sil = silhouette_score(X_scaled_df[mask_valid], db_labels[mask_valid])

    # SECTION 10: LABELS
    ph_sorted   = cluster_means['ph'].sort_values()
    cluster_ids = ph_sorted.index.tolist()
    LABEL_MAP   = {
        cluster_ids[0]: ('Polluted Water',   'card-polluted', 'polluted', '🔴'),
        cluster_ids[1]: ('Moderate Quality', 'card-moderate', 'moderate', '🟡'),
        cluster_ids[2]: ('Clean Water',      'card-clean',    'clean',    '🟢'),
    }

    return {
        'df_raw': df_raw, 'df_clean': df_clean, 'features': features,
        'labels_true': labels_true, 'X_scaled': X_scaled_df,
        'scaler': scaler, 'X_pca': X_pca, 'pca': pca,
        'K_range': list(K_range), 'inertias': inertias, 'silhouettes': silhouettes,
        'kmeans': kmeans, 'km_labels': km_labels, 'km_sil': km_sil,
        'cluster_means': cluster_means,
        'db_labels': db_labels, 'n_clusters_db': n_clusters_db,
        'n_noise': n_noise, 'db_sil': db_sil,
        'LABEL_MAP': LABEL_MAP, 'BEST_K': k,
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 💧 Water Quality Clustering")
    st.markdown("*Applied ML — *")
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload water_potability.csv", type=['csv'])

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    k_value     = st.slider("K-Means — k", 2, 8, 3)
    eps_value   = st.slider("DBSCAN — eps", 0.5, 5.0, 2.5, 0.1)
    min_samples = st.slider("DBSCAN — min_samples", 3, 20, 10)

    st.markdown("---")
    st.markdown("### 👥 Team")
    for name, role in [
        ("Marwan", "Data Loading + EDA"),
        ("Fares",  "Cleaning + Scaling"),
        ("Seif",   "PCA + K-Means"),
        ("Adham",  "DBSCAN + Visuals"),
        ("Omar",   "Evaluation + Report"),
    ]:
        st.markdown(f"**{name}** — {role}")


# ============================================================
# HEADER
# ============================================================
st.markdown("# 💧 Water Quality Clustering")
st.markdown("**Unsupervised Machine Learning | Applied ML**")
st.markdown("---")


# ============================================================
# RUN PIPELINE
# ============================================================
with st.spinner("Running pipeline..."):
    R = run_pipeline(uploaded_file, k=k_value, eps=eps_value, min_samples=min_samples)

df_raw        = R['df_raw']
df_clean      = R['df_clean']
features      = R['features']
labels_true   = R['labels_true']
X_scaled      = R['X_scaled']
scaler        = R['scaler']
X_pca         = R['X_pca']
pca           = R['pca']
K_range       = R['K_range']
inertias      = R['inertias']
silhouettes   = R['silhouettes']
kmeans        = R['kmeans']
km_labels     = R['km_labels']
km_sil        = R['km_sil']
cluster_means = R['cluster_means']
db_labels     = R['db_labels']
n_clusters_db = R['n_clusters_db']
n_noise       = R['n_noise']
db_sil        = R['db_sil']
LABEL_MAP     = R['LABEL_MAP']
BEST_K        = R['BEST_K']


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset",
    "🔍 EDA",
    "🤖 Clustering",
    "📈 Evaluation",
    "🔬 Predict",
])


# ────────────────────────────────────────────────────────────
# TAB 1 — DATASET  (Marwan)
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 📊 Dataset Overview")
    st.caption("Marwan — Section 1: Data Loading")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples",        f"{df_raw.shape[0]:,}")
    c2.metric("Features",       df_raw.shape[1] - 1)
    c3.metric("Missing Values", int(df_raw.isnull().sum().sum()))
    if labels_true is not None:
        c4.metric("Safe Water %", f"{labels_true.mean()*100:.0f}%")

    st.markdown("---")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("### Raw Data — first 10 rows")
        st.dataframe(df_raw.head(10), use_container_width=True)
    with col_r:
        st.markdown("### Missing Values")
        miss = df_raw.isnull().sum().reset_index()
        miss.columns = ['Feature', 'Missing']
        miss['%'] = (miss['Missing'] / len(df_raw) * 100).round(1)
        miss = miss[miss['Missing'] > 0]
        if len(miss):
            st.dataframe(miss, use_container_width=True)
        else:
            st.success("No missing values!")

    st.markdown("### Statistical Summary")
    st.dataframe(df_raw[features].describe().round(2), use_container_width=True)
    st.info("⚠️ Potability column excluded from training — Unsupervised Learning.")


# ────────────────────────────────────────────────────────────
# TAB 2 — EDA  (Marwan)
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.caption("Marwan — Section 4: EDA")

    # Feature distributions
    st.markdown("### Feature Distributions")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Feature Distributions (After Cleaning)',
                 fontsize=15, fontweight='bold', color=TEXT, y=1.01)
    for ax, col in zip(axes.flat, features):
        ax.hist(df_clean[col], bins=40, color='#00d4ff', alpha=0.75, edgecolor='#003344')
        ax.set_facecolor(BG)
        ax.set_title(col, fontweight='bold', fontsize=10, color=TEXT)
        ax.set_xlabel('Value', color='#9090b0')
        ax.set_ylabel('Count', color='#9090b0')
        ax.tick_params(colors='#9090b0')
        ax.grid(True, alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a3d5c')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    corr = df_clean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', linewidths=0.5, ax=ax,
                annot_kws={'size': 8, 'color': 'white'},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold',
                 color=TEXT, pad=15)
    ax.tick_params(colors='#9090b0')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.success("✅ Most features near-zero correlation — all 9 used in the model.")


# ────────────────────────────────────────────────────────────
# TAB 3 — CLUSTERING  (Seif + Adham)
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🤖 Clustering")

    # ── K-MEANS (Seif) ──
    st.markdown("### 🔵 K-Means Clustering")
    st.caption("Seif — Section 6 + 7: Optimal K + K-Means")

    with st.expander("📉 Elbow Method + Silhouette Score", expanded=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BG)
        fig.suptitle('Optimal K Selection', fontsize=14, fontweight='bold', color=TEXT)
        for ax in [ax1, ax2]:
            ax.set_facecolor(BG)
            ax.tick_params(colors='#9090b0')
            ax.grid(True, alpha=0.4)
            for sp in ax.spines.values(): sp.set_edgecolor('#3a3d5c')

        ax1.plot(K_range, inertias, 'o-', color='#00d4ff', linewidth=2.5, markersize=8)
        ax1.axvline(x=BEST_K, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Chosen k={BEST_K}')
        ax1.set_title('Elbow Method (Inertia)', fontweight='bold', color=TEXT)
        ax1.set_xlabel('Number of Clusters (k)', color='#9090b0')
        ax1.set_ylabel('Inertia', color='#9090b0')
        ax1.legend()

        ax2.plot(K_range, silhouettes, 's-', color='#43e97b', linewidth=2.5, markersize=8)
        ax2.axvline(x=BEST_K, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Chosen k={BEST_K}')
        ax2.set_title('Silhouette Scores', fontweight='bold', color=TEXT)
        ax2.set_xlabel('Number of Clusters (k)', color='#9090b0')
        ax2.set_ylabel('Silhouette Score', color='#9090b0')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters (k)", BEST_K)
    c2.metric("Silhouette Score", f"{km_sil:.4f}")
    c3.metric("Inertia", f"{kmeans.inertia_:,.0f}")

    # K-Means PCA scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for c in range(BEST_K):
        mask = km_labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=PALETTE[c], label=f'Cluster {c}', alpha=0.6, s=18, edgecolors='none')
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               c='white', s=200, marker='*', zorder=5,
               edgecolors='black', linewidths=0.8, label='Centroids')
    ax.set_title(f'K-Means Clustering (k={BEST_K}) — PCA 2D', fontsize=13, fontweight='bold', color=TEXT)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', color='#9090b0')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', color='#9090b0')
    ax.tick_params(colors='#9090b0')
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values(): sp.set_edgecolor('#3a3d5c')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Cluster profiles
    st.markdown("### Cluster Profiles")
    cm_display = cluster_means.copy().round(2)
    cm_display['Label'] = cm_display.index.map(lambda x: LABEL_MAP[x][0] if x in LABEL_MAP else '—')
    st.dataframe(cm_display, use_container_width=True)

    for cid, (lbl, css, _, emoji) in LABEL_MAP.items():
        row = cluster_means.loc[cid]
        st.markdown(f"""
        <div class='{css}'>
        <p><b>{emoji} {lbl} — Cluster {cid}</b></p>
        <p>pH = {row['ph']:.2f} &nbsp;|&nbsp;
           Turbidity = {row['Turbidity']:.2f} &nbsp;|&nbsp;
           Sulfate = {row['Sulfate']:.2f} &nbsp;|&nbsp;
           Organic Carbon = {row['Organic_carbon']:.2f}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── DBSCAN (Adham) ──
    st.markdown("### 🔴 DBSCAN Clustering")
    st.caption("Adham — Section 8: DBSCAN")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters Found", n_clusters_db)
    c2.metric("Noise Points",   n_noise)
    c3.metric("Noise %",        f"{n_noise/len(db_labels)*100:.1f}%")

    # DBSCAN PCA scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    unique_db = sorted(set(db_labels))
    lbl_map_db = {-1: 'Noise (−1)'}
    for i, c in enumerate(unique_db):
        if c != -1: lbl_map_db[c] = f'Cluster {c}'
    for i, c in enumerate(unique_db):
        mask  = db_labels == c
        color = NOISE_CLR if c == -1 else PALETTE4[i % len(PALETTE4)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=lbl_map_db[c],
                   alpha=0.5 if c != -1 else 0.2,
                   s=12 if c != -1 else 6, edgecolors='none')
    ax.set_title('DBSCAN Clustering — PCA 2D', fontsize=13, fontweight='bold', color=TEXT)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', color='#9090b0')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', color='#9090b0')
    ax.tick_params(colors='#9090b0')
    ax.legend(framealpha=0.3, markerscale=2)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values(): sp.set_edgecolor('#3a3d5c')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    if n_clusters_db <= 1:
        st.warning("⚠️ DBSCAN found only 1 cluster — data too dense. Try adjusting eps or min_samples.")

    st.markdown("---")

    # Side-by-side comparison
    st.markdown("### K-Means vs DBSCAN — Comparison")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle('K-Means vs DBSCAN — PCA 2D Comparison', fontsize=14, fontweight='bold', color=TEXT)

    for c in range(BEST_K):
        mask = km_labels == c
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=PALETTE[c], label=f'Cluster {c}', alpha=0.55, s=15, edgecolors='none')
    ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                c='white', s=200, marker='*', zorder=5, edgecolors='black', linewidths=0.8)
    ax1.set_title(f'K-Means  (k={BEST_K})  |  Sil={km_sil:.3f}', fontweight='bold', fontsize=11, color=TEXT)
    ax1.set_facecolor(BG); ax1.set_xlabel('PC1', color='#9090b0'); ax1.set_ylabel('PC2', color='#9090b0')
    ax1.tick_params(colors='#9090b0'); ax1.legend(framealpha=0.3); ax1.grid(True, alpha=0.3)

    for i, c in enumerate(unique_db):
        mask  = db_labels == c
        color = NOISE_CLR if c == -1 else PALETTE4[i % len(PALETTE4)]
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=lbl_map_db[c],
                    alpha=0.5 if c != -1 else 0.15, s=12, edgecolors='none')
    sil_txt = f"Sil={db_sil:.3f}" if db_sil else "Sil=N/A"
    ax2.set_title(f'DBSCAN  ({n_clusters_db} clusters, {n_noise} noise)  |  {sil_txt}',
                  fontweight='bold', fontsize=11, color=TEXT)
    ax2.set_facecolor(BG); ax2.set_xlabel('PC1', color='#9090b0'); ax2.set_ylabel('PC2', color='#9090b0')
    ax2.tick_params(colors='#9090b0'); ax2.legend(framealpha=0.3, markerscale=2); ax2.grid(True, alpha=0.3)

    for ax in [ax1, ax2]:
        for sp in ax.spines.values(): sp.set_edgecolor('#3a3d5c')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ────────────────────────────────────────────────────────────
# TAB 4 — EVALUATION  (Omar)
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 📈 Evaluation & Interpretation")
    st.caption("Omar — Section 10 + 11: Evaluation + Report")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**K-Means**")
        st.metric("Silhouette Score", f"{km_sil:.4f}")
        st.success(f"✅ Winner — {BEST_K} balanced clusters")
    with c2:
        st.markdown("**DBSCAN**")
        st.metric("Clusters Found", n_clusters_db)
        st.metric("Noise Points",   n_noise)
        st.error("❌ Not suitable — data too dense") if n_clusters_db <= 1 else st.info(f"ℹ️ {n_clusters_db} clusters")

    st.markdown("---")

    # Feature distributions per cluster
    st.markdown("### Feature Distributions per Cluster")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Feature Distributions per K-Means Cluster', fontsize=13, fontweight='bold', color=TEXT)
    for ax, col in zip(axes.flat, features):
        ax.set_facecolor(BG)
        for c in range(BEST_K):
            vals = df_clean.loc[km_labels == c, col]
            ax.hist(vals, bins=30, color=PALETTE[c], alpha=0.55, label=f'C{c}', edgecolor='none')
        ax.set_title(col, fontweight='bold', fontsize=9, color=TEXT)
        ax.set_xlabel('Value', fontsize=8, color='#9090b0')
        ax.tick_params(colors='#9090b0', labelsize=7)
        ax.legend(fontsize=7, framealpha=0.3)
        ax.grid(True, alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor('#3a3d5c')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Cluster heatmap
    st.markdown("### Cluster Profiles Heatmap")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    scaled_means = pd.DataFrame(
        scaler.transform(cluster_means),
        index=[f'Cluster {i}' for i in range(BEST_K)],
        columns=features
    )
    sns.heatmap(scaled_means, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=0.5, ax=ax, center=0,
                annot_kws={'color': 'black', 'size': 9},
                cbar_kws={'label': 'Standardised Mean'})
    ax.set_title('Cluster Profiles — Standardised Feature Means',
                 fontsize=12, fontweight='bold', color=TEXT)
    ax.tick_params(colors='#9090b0')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("### Final Report")
    sil_db_str = f"{db_sil:.4f}" if db_sil else "N/A"
    st.code(f"""
╔══════════════════════════════════════════════════════════════╗
║        WATER QUALITY CLUSTERING — FINAL REPORT               ║
╠══════════════════════════════════════════════════════════════╣
║  Dataset  : {df_raw.shape[0]:,} samples × {len(features)} features                    ║
║  Cleaning : Median imputation + IQR×3 outlier capping        ║
║  Scaling  : StandardScaler (mean=0, std=1)                   ║
║  PCA      : {sum(pca.explained_variance_ratio_)*100:.1f}% variance explained (2 components)        ║
╠══════════════════════════════════════════════════════════════╣
║  K-Means  │ k={BEST_K}  │ Silhouette = {km_sil:.4f}                    ║
║  DBSCAN   │ {n_clusters_db} cluster(s) │ Silhouette = {sil_db_str}                ║
╠══════════════════════════════════════════════════════════════╣
║  WINNER: K-Means                                             ║
╚══════════════════════════════════════════════════════════════╝
    """)


# ────────────────────────────────────────────────────────────
# TAB 5 — PREDICT
# ────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## 🔬 Predict a Water Sample")
    st.markdown("Enter the chemical values and the model will classify the water quality.")

    defaults = {
        'ph': (0.0, 14.0, 7.0), 'Hardness': (47.0, 324.0, 196.0),
        'Solids': (320.0, 61228.0, 22000.0), 'Chloramines': (0.3, 13.1, 7.1),
        'Sulfate': (129.0, 481.0, 333.0), 'Conductivity': (181.0, 754.0, 426.0),
        'Organic_carbon': (2.2, 28.3, 14.3), 'Trihalomethanes': (0.7, 124.0, 66.0),
        'Turbidity': (1.4, 6.7, 4.0),
    }
    who = {
        'ph': '6.5 – 8.5', 'Hardness': '< 300 mg/L', 'Solids': '< 500 mg/L',
        'Chloramines': '< 4 mg/L', 'Sulfate': '< 250 mg/L',
        'Conductivity': '< 400 μS/cm', 'Organic_carbon': '< 2 mg/L',
        'Trihalomethanes': '< 80 μg/L', 'Turbidity': '< 5 NTU',
    }

    c1, c2, c3 = st.columns(3)
    cols3  = [c1, c2, c3]
    inputs = {}
    for i, feat in enumerate(features):
        mn, mx, default = defaults.get(feat, (0.0, 100.0, 50.0))
        inputs[feat] = cols3[i % 3].number_input(
            feat, min_value=float(mn), max_value=float(mx),
            value=float(default), step=0.01
        )

    st.markdown("---")

    if st.button("🔍 Classify This Water Sample"):
        sample        = np.array([[inputs[f] for f in features]])
        sample_scaled = scaler.transform(sample)
        cluster_id    = kmeans.predict(sample_scaled)[0]

        if cluster_id in LABEL_MAP:
            lbl, _, kind, emoji = LABEL_MAP[cluster_id]
        else:
            lbl, kind, emoji = 'Moderate Quality', 'moderate', '🟡'

        css = f'result-{kind}'
        st.markdown(f"""
        <div class='{css}'>
            <h1 style='font-size:3rem'>{emoji}</h1>
            <h2 style='color:#e0e0e0'>{lbl}</h2>
            <p>Assigned to <b>Cluster {cluster_id}</b></p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Your Sample vs Cluster Average")
        cl_avg = df_clean[km_labels == cluster_id][features].mean()
        comp   = pd.DataFrame({
            'Your Sample':     [round(inputs[f], 3) for f in features],
            'Cluster Average': cl_avg.round(3).values,
            'WHO Safe Range':  [who.get(f, '—') for f in features],
        }, index=features)
        st.dataframe(comp, use_container_width=True)
