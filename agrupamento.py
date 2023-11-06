import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv('./Base/diabetes.csv')

scaler = StandardScaler()
df2 = df.copy()
df1 = df.copy()
df1[df1.columns[:-1]] = scaler.fit_transform(df1[df1.columns[:-1]])
df2[df2.columns[:-1]] = scaler.transform(df2[df2.columns[:-1]])

Q1 = df2[df2.columns[:-1]].quantile(0.25)
Q3 = df2[df2.columns[:-1]].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df2[df2.columns[:-1]] < (Q1 - 1.5 * IQR)) | (df2[df2.columns[:-1]] > (Q3 + 1.5 * IQR))).any(axis=1)

df1 = df1[~outliers]

numero_de_clusters = 2

algorithms = [
    KMeans(n_clusters=numero_de_clusters, random_state=0, n_init=10),
    AgglomerativeClustering(n_clusters=numero_de_clusters),
    DBSCAN(),
    MeanShift(),
    GaussianMixture(n_components=numero_de_clusters, random_state=0, n_init=10)
]

best_algorithm = None
best_metrics = None
best_metrics_mean = float('-inf')

def calculate_metrics(df, labels):
    if len(np.unique(labels)) < 2:
        return 0.0, float('inf'), 0.0 

    silhouette = silhouette_score(df[df.columns[:-1]], labels)
    davies_bouldin = davies_bouldin_score(df[df.columns[:-1]], labels)
    calinski_harabasz = calinski_harabasz_score(df[df.columns[:-1]], labels)
    return silhouette, davies_bouldin, calinski_harabasz

metrics_data = []
for algorithm in algorithms:
    labels = algorithm.fit_predict(df1[df1.columns[:-1]])
    silhouette, davies_bouldin, calinski_harabasz = calculate_metrics(df1, labels)
    
    metrics_mean = (silhouette + davies_bouldin + calinski_harabasz) / 3
    metrics_data.append({
        'Algoritmo': str(algorithm),
        'Silhouette': silhouette,
        'Davies-Bouldin': davies_bouldin,
        'Calinski-Harabasz': calinski_harabasz,
        'Média das Métricas': metrics_mean
    })
    if metrics_mean > best_metrics_mean:
        best_metrics_mean = metrics_mean
        best_algorithm = algorithm
        best_metrics = {
            'Silhouette': silhouette,
            'Davies-Bouldin': davies_bouldin,
            'Calinski-Harabasz': calinski_harabasz
        }

print("Melhor algoritmo de agrupamento:")
print(best_algorithm)
print("Métricas do melhor algoritmo:")
for metric, value in best_metrics.items():
    print(f"{metric}: {value}")

best_algorithm = KMeans(n_clusters=numero_de_clusters, random_state=0, n_init=10)
df1['BestAlgorithm'] = best_algorithm.fit_predict(df1[df1.columns[:-1]])

grouped_data = df1.groupby('BestAlgorithm')
statistics = []
for group_name, group_data in grouped_data:
    statistics.append(group_data.describe())

statistics_df = pd.concat(statistics)

print("\nEstatísticas Descritivas:")
print(statistics_df)

metrics_df = pd.DataFrame(metrics_data)
print("\nTabela de Métricas de Avaliação:")
print(metrics_df)
