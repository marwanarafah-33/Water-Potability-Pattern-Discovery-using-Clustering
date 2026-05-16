# Water Quality Clustering

so basically for our Applied ML project we decided to do something with water quality data. the idea is simple — we have a dataset with 3276 water samples, each one has 9 chemical measurements like pH, turbidity, sulfate levels etc. and we wanted to see if we could group them into clusters based on how similar they are, without actually telling the model which ones are clean or polluted.

no labels. no cheating. just the numbers.

---

## the idea

we used unsupervised learning (clustering) to let the algorithm figure out the groups on its own. after it groups them, we look at the chemical values in each group and interpret what they mean — that part is us, not the model.

turned out the data splits pretty naturally into 3 groups that we interpreted as clean, moderate, and polluted water based on the pH and turbidity values mostly.

---

## dataset

3276 water samples, 9 features:

- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity

there's also a Potability column (0 or 1) but we didn't use it for training — that's the whole point of unsupervised learning.

dataset from Kaggle → https://www.kaggle.com/datasets/adityakadiwal/water-potability

---

## what we actually did

cleaning — there were 1434 missing values across 3 columns (pH had 491 missing, Sulfate had 781). filled them with the median. also capped outliers using IQR × 3.

scaling — used StandardScaler so that features with big numbers (like Solids which goes up to 61,000) don't dominate over features with small numbers (like pH which is 0–14).

PCA — reduced the 9 features down to 2 dimensions so we could actually visualize the clusters on a 2D plot.

finding optimal k — used Elbow Method and Silhouette Score, both pointed to k=3.

K-Means — ran it with k=3, got a silhouette score of 0.077 (low because water quality data naturally overlaps, there's no sharp boundary between clean and polluted).

DBSCAN — tried it as a second model for comparison. found only 1 cluster + 67 noise points. not suitable for this kind of data.

---

## user interface

we built a full interactive web app using Streamlit so anyone can explore the results without touching the code.

the app has 5 tabs:
- Dataset → shows the raw data, missing values, and statistics
- EDA → histograms for all 9 features + correlation heatmap
- Clustering → runs K-Means and/or DBSCAN with PCA scatter plots
- Evaluation → silhouette score, cluster heatmap, model comparison
- Predict Sample → enter chemical values and get a water quality prediction

the sidebar lets you upload your own CSV, choose between K-Means / DBSCAN / Both, and adjust model settings like k, eps, and min_samples.

---

## how to run it

git clone https://github.com/your-username/water-quality-clustering.git
cd water-quality-clustering
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
streamlit run water_quality_app.py

make sure water_potability.csv is in the same folder. the app will open automatically in your browser at localhost:8501.

⚠️ don't run it with python water_quality_app.py — it won't work. always use streamlit run.

---

## project files

water-quality-clustering/
├── water_quality_app.py    ← the whole project (single file)
├── water_potability.csv    ← dataset (download from Kaggle)
└── README.md

---

## team

Marwan → Data Loading + EDA
Fares → Data Cleaning + Scaling
Seif → PCA + Optimal K + K-Means
Adham → DBSCAN + Visualizations
Omar → Evaluation + Interpretation + Report
