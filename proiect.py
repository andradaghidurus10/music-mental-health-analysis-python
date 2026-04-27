import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Clusterizare pe scorurile de sanatate mintala:
# Anxiety, Depression, Insomnia, OCD

# Algoritmul KMeans
# -> algoritm de invatare nesupravegheata folosit pt clusterizare
# -> imparte datele in K grupuri (clustere)
# -> a.i elementele din acelasi grup sa fie cat mai asemanatoare intre ele


# Idee de baza:
# "cum pot sa impart datele mele in K grupuri a.i. punctele apropiate sa fie in acelasi grup"

# foloseste distante intre puncte

# ----------------------------------

# df -> data frame
# tabel de date in python care arata exact ca un excel
df = pd.read_csv(r"D:\Andrada NEW\PythonProject2\bd_muzica - Tabel complet.csv")
print(df.head())
print()

print(df.isnull().sum())

print()
df.info()
print()

print(df.columns)
print()

# alegem doar coloanele pe care vrem sa facem clusterizare
coloane_clusterizare = ["Anxiety", "Depression", "Insomnia", "OCD"]

print("Coloanele folosite pentru clusterizare sunt:")
print(coloane_clusterizare)
print()

# --------------- clusterizare :

# creare matrice de intrare pt algoritm
x = df[coloane_clusterizare]

print(x.head())  # verificare ca matricea este corecta
# verificam ca sunt doar numere si ca nu exista text

# ------------- tratarea valorilor lipsa:

# daca exista valori lipsa, le completam cu media coloanei
imputer = SimpleImputer(strategy="mean")
x = imputer.fit_transform(x)


# ------------- standardizare:

# standardizarea: transformarea valorilor astfel incat coloanele sa fie comparabile
# formula: (valoare-media coloanei)/deviatia standard
# asta face ca fiecare coloana sa fie adusa la cam aceeasi scara

# de ce avem nevoie ?
# -> algoritmul KMeans foloseste distante
# daca o coloana are valori mai mari si alta coloana are valori mai mici, coloana mare va domina

# creare obiect de standardizare:
# pregatesc mecanismul care va aduce variabilele pe aceeasi scara
scaler = StandardScaler()

# scalerul se uita la fiecare coloana si invata :
# 1. media
# 2. deviatia standard

# apoi transforma valorile pe baza lor
x_scaler = scaler.fit_transform(x)  # rezultatul transformarii
# transforma datele intr-un numpy array
# numpy array -> tabel numeric simplu fara nume de coloane


# incepere algoritm KMeans

# calculare nr de clustere dupa metoda Elbow
# algoritmul testeaza care este varianta cea mai buna (ia pe rand k = 1,2,3,...)
# graficul arata unde apare "cotul"
# astfel alegem n_clusters = valoare corecta

inertii = []  # punem in ea valorile de inertie pt fiecare nr testat

for k in range(1, 17):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x_scaler)
    inertii.append(model.inertia_)

plt.plot(range(1, 17), inertii)
plt.xlabel("Nr clustere")
plt.ylabel("Inertia")
plt.title("Metoda Elbow")
plt.show()

# dupa ce vezi graficul, alegi numarul de clustere
# momentan punem 3 ca exemplu
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaler)
df["Cluster"] = clusters

print(df["Cluster"].value_counts())
print()

# media variabilelor pe fiecare cluster
print(df.groupby("Cluster")[coloane_clusterizare].mean())
print()

# PCA pentru vizualizare 2D
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
print()

print(df.groupby("Cluster")[coloane_clusterizare].mean().T)

# silhouette score:
scor = silhouette_score(x_scaler, clusters)
print("Scorul silhouette =", scor)




#----------------------------------------------------------------------------------------#


df = pd.read_csv(r"D:\Andrada NEW\PythonProject2\bd_muzica - Tabel complet.csv")
print(df.head())
print()

print(df.isnull().sum())

print()
df.info()
print()

print(df.columns)
print()

# transformam in numere coloana Music effects
# Improve = 2, No effect = 1, Worsen = 0
df["Music effects"] = df["Music effects"].map({
    "Improve": 2,
    "No effect": 1,
    "Worsen": 0
})

print("Valorile unice din Music effects dupa transformare:")
print(df["Music effects"].unique())
print()

# alegem doar coloanele pe care vrem sa facem clusterizare
coloane_clusterizare = ["Hours per day", "Music effects", "Anxiety", "Depression"]

print("Coloanele folosite pentru clusterizare sunt:")
print(coloane_clusterizare)
print()

# --------------- clusterizare :

# creare matrice de intrare pt algoritm
x = df[coloane_clusterizare]

print(x.head())  # verificare ca matricea este corecta
# verificam ca sunt doar numere si ca nu exista text

# ------------- tratarea valorilor lipsa:

# daca exista valori lipsa, le completam cu media coloanei
imputer = SimpleImputer(strategy="mean")
x = imputer.fit_transform(x)

# ------------- standardizare:

# standardizarea: transformarea valorilor astfel incat coloanele sa fie comparabile
# formula: (valoare-media coloanei)/deviatia standard
# asta face ca fiecare coloana sa fie adusa la cam aceeasi scara

# de ce avem nevoie ?
# -> algoritmul KMeans foloseste distante
# daca o coloana are valori mai mari si alta coloana are valori mai mici, coloana mare va domina

# creare obiect de standardizare:
# pregatesc mecanismul care va aduce variabilele pe aceeasi scara
scaler = StandardScaler()

# scalerul se uita la fiecare coloana si invata :
# 1. media
# 2. deviatia standard

# apoi transforma valorile pe baza lor
x_scaler = scaler.fit_transform(x)  # rezultatul transformarii
# transforma datele intr-un numpy array
# numpy array -> tabel numeric simplu fara nume de coloane

# incepere algoritm KMeans

# calculare nr de clustere dupa metoda Elbow
# algoritmul testeaza care este varianta cea mai buna (ia pe rand k = 1,2,3,...)
# graficul arata unde apare "cotul"
# astfel alegem n_clusters = valoare corecta

inertii = []  # punem in ea valorile de inertie pt fiecare nr testat

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(x_scaler)
    inertii.append(model.inertia_)

plt.plot(range(1, 11), inertii)
plt.xlabel("Nr clustere")
plt.ylabel("Inertia")
plt.title("Metoda Elbow")
plt.show()

# alegem 3 clustere pe baza graficului Elbow
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaler)
df["Cluster"] = clusters

print(df["Cluster"].value_counts())
print()

# media variabilelor pe fiecare cluster
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
print()

print(df.groupby("Cluster")[coloane_clusterizare].mean().T)

scor = silhouette_score(x_scaler, clusters)
print("Scorul silhouette =", scor)
