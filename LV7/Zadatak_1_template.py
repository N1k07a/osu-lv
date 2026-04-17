import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()


#-----zadatak 1.1--------
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
 
opisi = [
    "Tip 1: 3 grupe (kompaktne)",
    "Tip 2: 3 grupe (transformirane)",
    "Tip 3: 4 grupe (razl. raspršenja)",
    "Tip 4: 2 grupe (kružnice)",
    "Tip 5: 2 grupe (polumjeseci)"
]
 
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    ax = axes[flagc - 1]
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7)
    ax.set_title(opisi[flagc - 1], fontsize=10)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')

plt.suptitle('Zadatak 1.1 – Prikaz svih tipova podataka', fontsize=13)
plt.tight_layout()
plt.show()

#----zadatak 1.2--------
optimalni_K = {1: 3, 2: 3, 3: 4, 4: 2, 5: 2}
 
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
 
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    K = optimalni_K[flagc]
 
    km = KMeans(n_clusters=K,
                init='k-means++',
                n_init=10,
                random_state=0)
    km.fit(X)
    labels = km.predict(X)
    centres = km.cluster_centers_
 
    ax = axes[flagc - 1]
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    ax.scatter(centres[:, 0], centres[:, 1],
               c='black', marker='X', s=200, zorder=5, label='Centri')
    ax.set_title(f'Tip {flagc} – K={K}', fontsize=10)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(fontsize=8)
 
axes[5].axis('off')
plt.suptitle('Zadatak 1.2 – K-means grupiranje (optimalni K)', fontsize=13)
plt.tight_layout()
plt.show()

# ── Posebno: prikaz utjecaja različitog K za Tip 1 ──────────────
X = generate_data(500, 1)
 
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, K in enumerate([2, 3, 4, 5]):
    km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=0)
    km.fit(X)
    labels = km.predict(X)
    centres = km.cluster_centers_
 
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    axes[i].scatter(centres[:, 0], centres[:, 1],
                    c='black', marker='X', s=150, zorder=5)
    axes[i].set_title(f'K = {K}')
    axes[i].set_xlabel('$x_1$')
    axes[i].set_ylabel('$x_2$')
 
plt.suptitle('Zadatak 1.2 – Utjecaj parametra K (Tip 1 podataka)', fontsize=12)
plt.tight_layout()
plt.show()
 
# ═══════════════════════════════════════════════════════════════════
# ZADATAK 1.3 – Elbow metoda (za svaki tip podataka)
# ═══════════════════════════════════════════════════════════════════
 
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
 
K_range = range(1, 11)
 
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    J_values = []
 
    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
        km.fit(X)
        J_values.append(km.inertia_)
 
    ax = axes[flagc - 1]
    ax.plot(K_range, J_values, 'o-', linewidth=2, markersize=5, color='steelblue')
    ax.set_title(f'Tip {flagc} – Elbow metoda', fontsize=10)
    ax.set_xlabel('Broj grupa K')
    ax.set_ylabel('J (inertia)')
    ax.set_xticks(list(K_range))

 
axes[5].axis('off')
plt.suptitle('Zadatak 1.3 – Elbow metoda za sve tipove podataka', fontsize=13)
plt.tight_layout()
plt.show()
 
# ═══════════════════════════════════════════════════════════════════
# KOMENTAR REZULTATA (ispis u konzoli)
# ══════════════════════════