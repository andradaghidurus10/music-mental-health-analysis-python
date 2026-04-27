
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ce tipuri de utilizatori exista in finctie de modul in care folosesc muzica in viata lor
# variabile: Hours per day , While working , Instrumentalist, Composer , Foreign languages


df = pd.read_csv(r"D:\Andrada NEW\PythonProject2\bd_muzica - Tabel complet.csv")

print(df.head())
print()

print(df.isnull().sum())
print()

df.info()
print()

print(df.columns)
print()

coloane_clusterizare = [
    "Hours per day",
    "While working",
    "Instrumentalist",
    "Composer",
    "Foreign languages"
]

print("Coloanele folosite pentru clusterizare sunt:")
print(coloane_clusterizare)
print()

x = df[coloane_clusterizare]

print(x.head())

imputer = SimpleImputer(strategy="mean")
x = imputer.fit_transform(x)

scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

inertii = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x_scaler)
    inertii.append(model.inertia_)

plt.plot(range(1, 11), inertii)
plt.xlabel("Nr clustere")
plt.ylabel("Inertia")
plt.title("Metoda Elbow")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaler)
df["Cluster"] = clusters

print(df["Cluster"].value_counts())
print()

print(df.groupby("Cluster")[coloane_clusterizare].mean())
print()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaler)

plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, cmap="viridis")
plt.title("Vizualizarea clusterelor folosind PCA")
plt.xlabel("Componenta principala 1")
plt.ylabel("Componenta principala 2")
plt.show()

print()
print()
print(df.groupby("Cluster")[coloane_clusterizare].mean().T)

scor = silhouette_score(x_scaler, clusters)
print("Scorul silhouette =", scor)


