import numpy as np
from PIL import Image

# Use a seed
np.random.seed(42)

# Open Image
image = np.array(Image.open("image.jpg"))
dtype = np.uint8

# Ajouter un bruit gaussien
sigma = 25
noise = np.random.normal(0, sigma, image.shape)
noisy_image = np.clip(image + noise, 0, 255).astype(dtype)
Image.fromarray(noisy_image).save("noisy_image.jpg")
