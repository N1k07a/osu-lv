import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ═══════════════════════════════════════════════════════════════════
# UČITAVANJE SLIKE (iz template-a)
# ═══════════════════════════════════════════════════════════════════

# ucitaj sliku  ← promijeni putanju prema svojoj slici
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transformiraj sliku u 2D numpy polje
# svaki red = RGB vrijednosti jednog piksela
w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))

# ═══════════════════════════════════════════════════════════════════
# ZADATAK 2.1 – Koliko različitih boja ima slika?
# ═══════════════════════════════════════════════════════════════════

unique_colors = np.unique(img_array, axis=0)
print("=" * 60)
print("ZADATAK 2.1 – Analiza boja originalne slike")
print("=" * 60)
print(f"Dimenzije slike:         {w} x {h} piksela")
print(f"Ukupan broj piksela:     {w * h}")
print(f"Broj različitih boja:    {len(unique_colors)}")
print()

# ═══════════════════════════════════════════════════════════════════
# ZADATAK 2.2 – Primjena K-means na RGB vrijednosti piksela
# ═══════════════════════════════════════════════════════════════════

K = 5   # broj boja (grupa) – mijenjaj po želji

print(f"Pokrećem K-means s K={K} grupama...")

km = KMeans(n_clusters=K,
            init='k-means++',
            n_init=5,
            random_state=42)
km.fit(img_array)

print(f"Konvergencija nakon {km.n_iter_} iteracija.")
print(f"J (inertia) = {km.inertia_:.4f}")
print()

# ═══════════════════════════════════════════════════════════════════
# ZADATAK 2.3 – Zamjena svakog piksela s centrom grupe
# ═══════════════════════════════════════════════════════════════════

labels = km.labels_             # indeks grupe za svaki piksel
centres = km.cluster_centers_   # RGB vrijednosti centara (K x 3)

# Svaki piksel dobiva RGB vrijednost svog centra
img_array_aprox = centres[labels]

# Pretvori natrag u 3D (w x h x 3) i u uint8 za prikaz
img_quantized = np.reshape(img_array_aprox, (w, h, d))

# ═══════════════════════════════════════════════════════════════════
# ZADATAK 2.4 – Usporedba originalne i kvantizirane slike
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(img)
axes[0].set_title(f"Originalna slika\n({len(unique_colors):,} boja)", fontsize=12)
axes[0].axis('off')

axes[1].imshow(img_quantized)
axes[1].set_title(f"Kvantizirana slika\n(K={K} boja)", fontsize=12)
axes[1].axis('off')

plt.suptitle('Zadatak 2.4 – Usporedba originalne i kvantizirane slike', fontsize=13)
plt.tight_layout()
plt.show()

# ── Prikaz za više vrijednosti K ─────────────────────────────────
K_values = [2, 4, 8, 16, 32]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

axes[0].imshow(img)
axes[0].set_title(f"Original\n({len(unique_colors):,} boja)")
axes[0].axis('off')

for i, k in enumerate(K_values):
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init=3, random_state=42)
    km_temp.fit(img_array)
    lbl = km_temp.labels_
    cen = km_temp.cluster_centers_
    approx = cen[lbl].reshape(w, h, d)

    axes[i + 1].imshow(approx)
    axes[i + 1].set_title(f"K = {k}")
    axes[i + 1].axis('off')

plt.suptitle('Zadatak 2.4 – Utjecaj parametra K na kvalitetu kvantizacije', fontsize=13)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# ZADATAK 2.6 – Elbow metoda (J u ovisnosti o K)
# ═══════════════════════════════════════════════════════════════════

print("Računam Elbow metodu (K = 1 do 15)...")

J_values = []
K_range = range(1, 16)

for k in K_range:
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init=3, random_state=42)
    km_temp.fit(img_array)
    J_values.append(km_temp.inertia_)
    print(f"  K={k:2d}  J = {km_temp.inertia_:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), J_values, 'o-', linewidth=2,
         markersize=6, color='steelblue')
plt.xlabel("Broj grupa K", fontsize=12)
plt.ylabel("J (inertia)", fontsize=12)
plt.title("Zadatak 2.6 – Elbow metoda za kvantizaciju slike", fontsize=13)
plt.xticks(list(K_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print()
print("KOMENTAR – Elbow metoda:")
print("  Tražimo K gdje J prestaje naglo padati.")
print("  Za fotografije lakat je često teško jasno detektirati")
print("  jer boje prelaze jednu u drugu bez oštrih granica.")
print()

# ═══════════════════════════════════════════════════════════════════
# ZADATAK 2.7 – Svaka grupa kao zasebna binarna slika
# ═══════════════════════════════════════════════════════════════════

# labels_2d[i,j] = indeks grupe piksela na poziciji (i,j)
labels_2d = labels.reshape(w, h)

fig, axes = plt.subplots(1, K + 1, figsize=(4 * (K + 1), 4))

# Lijevo: originalna slika za referencu
axes[0].imshow(img)
axes[0].set_title("Originalna slika", fontsize=10)
axes[0].axis('off')

for k in range(K):
    # Binarna slika: 1 (bijela) tamo gdje piksel pripada grupi k
    binary = (labels_2d == k).astype(np.float64)

    # Boja centra te grupe (za naslov)
    r, g, b = centres[k]

    axes[k + 1].imshow(binary, cmap='gray', vmin=0, vmax=1)
    axes[k + 1].set_title(
        f"Grupa {k + 1}\nRGB=({r:.2f},{g:.2f},{b:.2f})", fontsize=8)
    axes[k + 1].axis('off')

plt.suptitle(
    'Zadatak 2.7 – Binarne slike po grupama\n'
    '(bijelo = pikseli koji pripadaju toj grupi)',
    fontsize=12)
plt.tight_layout()
plt.show()

print()
print("KOMENTAR – Binarne slike:")
print("  Svaka binarna slika prikazuje segment slike.")
print("  Npr. jedna grupa može odgovarati pozadini,")
print("  druga tekstu, treća objektima na slici.")
print("  Ovo je zapravo SEGMENTACIJA SLIKE primjenom K-means!")
print()

# ═══════════════════════════════════════════════════════════════════
# PRIKAZ BOJA CENTARA (bonus – pregled pronađenih boja)
# ═══════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(1, 1, figsize=(8, 1.5))
for k in range(K):
    rect = plt.Rectangle([k / K, 0], 1 / K, 1,
                         color=centres[k])
    ax.add_patch(rect)
    ax.text(k / K + 0.5 / K, 0.5,
            f"G{k + 1}\n({centres[k][0]:.2f},{centres[k][1]:.2f},{centres[k][2]:.2f})",
            ha='center', va='center', fontsize=8,
            color='white' if np.mean(centres[k]) < 0.5 else 'black')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title(f'Pronađene boje centara (K={K})', fontsize=11)
plt.tight_layout()
plt.show()
