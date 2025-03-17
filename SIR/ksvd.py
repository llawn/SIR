import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from PIL import Image
from scipy.linalg import svd
import matplotlib.pyplot as plt
from patches import extract_patches, recontruct_image

# seed
np.random.seed(42)
image = np.array(Image.open("image.jpg")) / 255.0 # Normaliser l'image
noisy_image = np.array(Image.open("noisy_image.jpg")) / 255.0 # Normaliser l'image

# parameters
patch_size = 8 # n = 64
n_atoms = 256 # Nombre d'atomes dans le dictionnaire
n_iterations = 10

# Dictionnaire
D = np.random.randn(patch_size * patch_size * 3, n_atoms)
D = D / np.linalg.norm(D, axis = 0)  # Normaliser les atomes

# K-SVD
for iteration in range(n_iterations):
    print(f"Iteration {iteration + 1}/{n_iterations}")
    # Codage parcimonieux avec OMP
    patches = extract_patches(noisy_image, patch_size)
    alpha = np.zeros((n_atoms, patches.shape[0]))
    for i, patch in enumerate(patches):
        omp = OMP(n_nonzero_coefs=n_iterations)
        omp.fit(D, patch)
        alpha[:, i] = omp.coef_

    # Mise à jour du dictionnaire avec SVD
    for k in range(n_atoms):
        # Trouver les patches qui utilisent l'atome k
        patch_indices = np.where(alpha[k, :] != 0)[0]
        if len(patch_indices) == 0:
            continue
        
        # Calculer le résidu pour ces patches
        E = patches[patch_indices].T - D @ alpha[:, patch_indices]
        E += D[:, k][:, np.newaxis] @ alpha[k, patch_indices][np.newaxis, :]
        
        # Appliquer SVD pour mettre à jour l'atome k
        U, S, Vt = svd(E, full_matrices=False)
        D[:, k] = U[:, 0]
        alpha[k, patch_indices] = S[0] * Vt[0, :]

# Reconstruire l'image débruitée
denoised_patches = D @ alpha
denoised_image = reconstruct_image(denoised_patches, noisy_image.shape, patch_size)

# Calculer le PSNR
def psnr(original, noisy):
    mse = np.mean((original - noisy) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse))

psnr_ksvd_color = psnr(image, denoised_image)
print(f"PSNR (K-SVD couleur globale): {psnr_ksvd_color:.2f} dB")

# Afficher les résultats
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Image bruitée")
plt.imshow(noisy_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Image débruitée (K-SVD couleur)")
plt.imshow(denoised_image)
plt.axis('off')
plt.show()
