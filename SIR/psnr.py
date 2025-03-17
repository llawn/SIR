import numpy as np
from math import log10
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Charge une image et la normalise entre 0 et 1.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image) / 255.0
    return image


def psnr(image_ref, image_comp):
    # Calculer l'erreur quadratique moyenne (MSE)
    mse = np.mean((image_ref - image_comp) ** 2)

    # Si le MSE est nul, le PSNR est infini (images identiques)
    if mse == 0:
        return float('inf')

    # Valeur maximale des pixels (1 pour les images normalisées)
    max_pixel = 1.0

    # Calculer le PSNR
    psnr_value = 10 * log10(max_pixel ** 2 / mse)
    return psnr_value


def main():
    # Chemins des images
    denoised_image_path = 'denoised_image.jpg'
    image_crop_path = 'image_crop.jpg'
    noisy_image_crop_path = 'noisy_image_crop.jpg'

    # Charger les images
    denoised_image = load_image(denoised_image_path)
    image_crop = load_image(image_crop_path)
    noisy_image_crop = load_image(noisy_image_crop_path)

    # Calculer le PSNR
    psnr_noisy = psnr(image_crop, noisy_image_crop)
    psnr_denoised = psnr(image_crop, denoised_image)

    # Afficher les résultats
    print(f"PSNR entre l'image originale et l'image bruitée : {psnr_noisy:.2f} dB")
    print(f"PSNR entre l'image originale et l'image débruîtée : {psnr_denoised:.2f} dB")

    # Afficher les images avec matplotlib
    plt.figure(figsize=(15, 5))

    # Afficher l'image originale
    plt.subplot(1, 3, 1)
    plt.imshow(image_crop)
    plt.title("Image Originale")
    plt.axis('off')

    # Afficher l'image bruitée
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image_crop)
    plt.title(f"Image Bruitée\nPSNR: {psnr_noisy:.2f} dB")
    plt.axis('off')

    # Afficher l'image débruîtée
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image)
    plt.title(f"Image Débruîtée\nPSNR: {psnr_denoised:.2f} dB")
    plt.axis('off')

    # Afficher la figure
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
