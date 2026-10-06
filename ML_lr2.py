# coding: utf-8
"""
Лабораторная работа №2: Кластеризация
Вариант 12
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift,
                             SpectralClustering, AgglomerativeClustering,
                             DBSCAN, OPTICS, Birch, BisectingKMeans)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.neighbors import NearestNeighbors

# ============================================================
# 0. ПАПКА ДЛЯ СОХРАНЕНИЯ ГРАФИКОВ
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Папка скрипта: {SCRIPT_DIR}")
print(f"Папка для результатов: {RESULTS_DIR}")


def save_fig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Сохранено: {name}")


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ (Вариант 12)
# ============================================================
print("\n" + "=" * 70)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

file_path = os.path.join(SCRIPT_DIR, 'Lab2_Dataset_Clustering.xlsx')

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Файл не найден: {file_path}\n"
        f"Положите Lab2_Dataset_Clustering.xlsx в папку: {SCRIPT_DIR}"
    )

print(f"Загружаю: {file_path}")
df = pd.read_excel(file_path, header=None)

# Вариант 12 -> столбцы AH, AI, AJ (индексы 33, 34, 35)
col_a, col_b, col_y = 33, 34, 35

data = df.iloc[3:, [col_a, col_b, col_y]].copy()
data.columns = ['a', 'b', 'y']
data = data.dropna()
data['a'] = pd.to_numeric(data['a'], errors='coerce')
data['b'] = pd.to_numeric(data['b'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data = data.dropna().reset_index(drop=True)

X = data[['a', 'b']].values
y_true = data['y'].values.astype(int)

print(f"Размер датасета: {X.shape}")
print(f"Уникальные метки: {np.unique(y_true)}")

unique_vals, counts = np.unique(y_true, return_counts=True)
dist = dict(zip(unique_vals.tolist(), counts.tolist()))
print(f"Распределение меток: {dist}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 2. ИСХОДНЫЕ ДАННЫЕ
# ============================================================
print("\n" + "=" * 70)
print("2. ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ")
print("=" * 70)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
                      c=y_true, cmap='viridis', s=50,
                      edgecolors='k', alpha=0.7)
plt.colorbar(scatter, label='Истинный класс')
plt.title('Исходные данные (Вариант 12)')
plt.xlabel('Признак a (стандартизованный)')
plt.ylabel('Признак b (стандартизованный)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('01_initial_data.png')
plt.show()

# ============================================================
# 3. МЕТОД ЛОКТЯ + СИЛУЭТ
# ============================================================
print("\n" + "=" * 70)
print("3. ПОИСК ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ")
print("=" * 70)

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Число кластеров')
axes[0].set_ylabel('Инерция (Distortion)')
axes[0].set_title('Метод локтя (Elbow Method)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), silhouettes, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Число кластеров')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Силуэтный коэффициент')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_fig('02_optimal_k.png')
plt.show()

# ============================================================
# 4. ПОДБОР ПАРАМЕТРОВ ДЛЯ DBSCAN, OPTICS, GMM
# ============================================================
print("\n" + "=" * 70)
print("4. ПОДБОР ПАРАМЕТРОВ (DBSCAN, OPTICS, GMM)")
print("=" * 70)

# --- 4.1. k-расстояние для DBSCAN ---
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(distances, linewidth=2)
plt.xlabel('Точки (отсортированные)')
plt.ylabel(f'Расстояние до {k}-го соседа')
plt.title('k-расстояние для подбора eps в DBSCAN')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.3, color='r', linestyle='--', label='eps = 0.3')
plt.axhline(y=0.25, color='g', linestyle='--', label='eps = 0.25')
plt.axhline(y=0.35, color='b', linestyle='--', label='eps = 0.35')
plt.legend()
plt.tight_layout()
save_fig('05_k_distance.png')
plt.show()

# --- 4.2. Подбор eps для DBSCAN ---
print("\nПодбор eps для DBSCAN (min_samples=5):")
best_eps, best_ari_db = None, -1
for eps in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
    db = DBSCAN(eps=eps, min_samples=5)
    lbl = db.fit_predict(X_scaled)
    n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
    n_noise = list(lbl).count(-1)
    if n_cl > 1:
        ari = adjusted_rand_score(y_true, lbl)
        if ari > best_ari_db:
            best_ari_db = ari
            best_eps = eps
        print(f"  eps={eps}: кластеров={n_cl:2d}, шум={n_noise:3d}, "
              f"ARI={ari:.4f}")
    else:
        print(f"  eps={eps}: кластеров={n_cl:2d}, шум={n_noise:3d}, ARI=N/A")

print(f"\nЛучший eps для DBSCAN: {best_eps} (ARI = {best_ari_db:.4f})")

# --- 4.3. Подбор xi для OPTICS ---
print("\nПодбор xi для OPTICS (min_samples=5):")
best_xi, best_ari_op = None, -1
for xi in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    opt = OPTICS(min_samples=5, xi=xi)
    lbl = opt.fit_predict(X_scaled)
    n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
    n_noise = list(lbl).count(-1)
    if n_cl > 1:
        ari = adjusted_rand_score(y_true, lbl)
        if ari > best_ari_op:
            best_ari_op = ari
            best_xi = xi
        print(f"  xi={xi}: кластеров={n_cl:2d}, шум={n_noise:3d}, "
              f"ARI={ari:.4f}")
    else:
        print(f"  xi={xi}: кластеров={n_cl:2d}, шум={n_noise:3d}, ARI=N/A")

print(f"\nЛучший xi для OPTICS: {best_xi} (ARI = {best_ari_op:.4f})")

# --- 4.4. Подбор covariance_type для GMM ---
print("\nПодбор covariance_type для GMM:")
best_cov, best_ari_gmm = None, -1
for cov in ['full', 'tied', 'diag', 'spherical']:
    gmm = GaussianMixture(n_components=3, covariance_type=cov,
                          random_state=42)
    lbl = gmm.fit_predict(X_scaled)
    ari = adjusted_rand_score(y_true, lbl)
    sil = silhouette_score(X_scaled, lbl)
    if ari > best_ari_gmm:
        best_ari_gmm = ari
        best_cov = cov
    print(f"  {cov:10s}: ARI={ari:.4f}, Silhouette={sil:.4f}")

print(f"\nЛучший covariance_type: {best_cov} (ARI = {best_ari_gmm:.4f})")

# ============================================================
# 5. АЛГОРИТМЫ (с лучшими параметрами)
# ============================================================
print("\n" + "=" * 70)
print("5. ПРИМЕНЕНИЕ АЛГОРИТМОВ КЛАСТЕРИЗАЦИИ")
print("=" * 70)

n_clusters = len(np.unique(y_true))
print(f"Число кластеров (по истинным меткам): {n_clusters}\n")

algorithms = {}

algorithms['K-Means'] = KMeans(
    n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)

algorithms['Affinity Propagation'] = AffinityPropagation(
    random_state=42, damping=0.9)

algorithms['MeanShift'] = MeanShift()

algorithms['Spectral Clustering'] = SpectralClustering(
    n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')

algorithms['Ward'] = AgglomerativeClustering(
    n_clusters=n_clusters, linkage='ward')

algorithms['Agglomerative'] = AgglomerativeClustering(
    n_clusters=n_clusters, linkage='complete')

# DBSCAN с лучшим eps
algorithms['DBSCAN'] = DBSCAN(
    eps=best_eps if best_eps else 0.3, min_samples=5)

try:
    import hdbscan
    algorithms['HDBSCAN'] = hdbscan.HDBSCAN(min_cluster_size=10)
    print("HDBSCAN загружен")
except ImportError:
    print("HDBSCAN не установлен - пропускаем")

# OPTICS с лучшим xi
algorithms['OPTICS'] = OPTICS(
    min_samples=5, xi=best_xi if best_xi else 0.15)

# GMM с лучшим covariance_type
algorithms['Gaussian Mixture'] = GaussianMixture(
    n_components=n_clusters,
    covariance_type=best_cov if best_cov else 'full',
    random_state=42)

algorithms['BIRCH'] = Birch(n_clusters=n_clusters)

algorithms['Bisecting K-Means'] = BisectingKMeans(
    n_clusters=n_clusters, random_state=42)

# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\n" + "=" * 70)
print("6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (цветные кластеры)")
print("=" * 70)

n_algs = len(algorithms)
n_cols = 4
n_rows = (n_algs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

results = {}

for idx, (name, algo) in enumerate(algorithms.items()):
    print(f"  -> {name}...")
    try:
        labels = algo.fit_predict(X_scaled)
        results[name] = labels

        if len(set(labels)) > 1 and -1 not in set(labels):
            sil = silhouette_score(X_scaled, labels)
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
        else:
            sil = ari = nmi = np.nan

        unique_labels = sorted(set(labels))
        for label in unique_labels:
            mask = labels == label
            if label == -1:
                axes[idx].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                                  c='black', s=30, marker='o',
                                  edgecolors='none', alpha=0.9)
            else:
                color = plt.cm.viridis(
                    label / max(1, len(unique_labels) - 1))
                axes[idx].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                                  c=[color], s=30, marker='o',
                                  edgecolors='k', linewidths=0.3, alpha=0.8)

        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        title = f'{name}\nКластеров: {n_clust}, Шум: {n_noise}'
        if not np.isnan(sil):
            title += f'\nSil: {sil:.3f} | ARI: {ari:.3f}'
        axes[idx].set_title(title, fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    except Exception as e:
        axes[idx].text(0.5, 0.5, f'{name}\nОшибка:\n{str(e)[:60]}',
                       ha='center', va='center',
                       transform=axes[idx].transAxes, fontsize=9)
        axes[idx].set_title(f'{name} (ошибка)')
        print(f"    Ошибка: {e}")

for idx in range(len(algorithms), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
save_fig('06_all_algorithms_tuned.png')
plt.show()

# ============================================================
# 7. СРАВНЕНИЕ МЕТРИК
# ============================================================
print("\n" + "=" * 70)
print("7. СРАВНЕНИЕ АЛГОРИТМОВ")
print("=" * 70)

comparison = []
for name, labels in results.items():
    n_clust = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    if n_clust > 1 and -1 not in set(labels):
        sil = silhouette_score(X_scaled, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
    else:
        sil = ari = nmi = np.nan

    comparison.append({
        'Алгоритм': name,
        'Кластеров': n_clust,
        'Шум': n_noise,
        'Silhouette': round(sil, 4) if not np.isnan(sil) else 'N/A',
        'ARI': round(ari, 4) if not np.isnan(ari) else 'N/A',
        'NMI': round(nmi, 4) if not np.isnan(nmi) else 'N/A'
    })

comparison_df = pd.DataFrame(comparison)

# Сортировка с приведением к числам
comparison_df['_ARI_sort'] = pd.to_numeric(comparison_df['ARI'],
                                           errors='coerce')
comparison_df = comparison_df.sort_values('_ARI_sort', ascending=False,
                                          na_position='last')
comparison_df = comparison_df.drop(columns='_ARI_sort').reset_index(drop=True)

print("\n" + comparison_df.to_string(index=False))

csv_path = os.path.join(RESULTS_DIR, 'comparison.csv')
comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nТаблица сохранена: {csv_path}")

# ============================================================
# 8. ВЫБРОСЫ
# ============================================================
print("\n" + "=" * 70)
print("8. АНАЛИЗ ВЫБРОСОВ")
print("=" * 70)

density_algs = [a for a in ['DBSCAN', 'OPTICS', 'HDBSCAN'] if a in results]

if density_algs:
    fig, axes = plt.subplots(1, len(density_algs),
                             figsize=(6 * len(density_algs), 6))
    if len(density_algs) == 1:
        axes = [axes]

    for idx, name in enumerate(density_algs):
        labels = results[name]
        unique_labels = sorted(set(labels))

        for label in unique_labels:
            mask = labels == label
            if label == -1:
                axes[idx].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                                  c='black', s=40, marker='o',
                                  edgecolors='none', alpha=0.9,
                                  label=f'Шум ({mask.sum()})')
            else:
                color = plt.cm.viridis(
                    label / max(1, len(unique_labels) - 1))
                axes[idx].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                                  c=[color], s=40, marker='o',
                                  edgecolors='k', linewidths=0.3,
                                  alpha=0.8,
                                  label=f'Кластер {label} ({mask.sum()})')

        n_noise = list(labels).count(-1)
        axes[idx].set_title(f'{name}\nВыбросов: {n_noise} '
                            f'({100 * n_noise / len(labels):.1f}%)')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('07_outliers_tuned.png')
    plt.show()

# ============================================================
# 9. ВЫВОДЫ
# ============================================================
print("\n" + "=" * 70)
print("9. ВЫВОДЫ")
print("=" * 70)

best_rows = comparison_df[comparison_df['ARI'] != 'N/A']
if len(best_rows) > 0:
    best_row = best_rows.iloc[0]
    print(f"\nЛучший алгоритм по ARI: {best_row['Алгоритм']}")
    print(f"   ARI = {best_row['ARI']}, NMI = {best_row['NMI']}, "
          f"Silhouette = {best_row['Silhouette']}")

print(f"\nЛучшие параметры:")
print(f"   DBSCAN eps      = {best_eps}  (ARI = {best_ari_db:.4f})")
print(f"   OPTICS xi       = {best_xi}   (ARI = {best_ari_op:.4f})")
print(f"   GMM covariance  = {best_cov}  (ARI = {best_ari_gmm:.4f})")

print(f"\nВсе результаты сохранены в: {RESULTS_DIR}")
print("=" * 70)