import numpy as np


def extract_patches(image: np.array, patch_size: int) -> np.array:
    patches = []
    for i in range(image.shape[0] - patch_size + 1):
        for j in range(image.shape[1] - patch_size + 1):
            patch = image[i:i+patch_size, j:j+patch_size, :].flatten()
            patches.append(patch)
    return np.array(patches)


def recontruct_image(patches: np.array, image_shape: tuple, patch_size: int) -> np.array:
    image = np.zeros(image_shape)
    count = np.zeros(image_shape)
    idx = 0
    for i in range(image_shape[0] - patch_size + 1):
        for j in range(image_shape[1] - patch_size + 1):
            image[i:i+patch_size, j:j+patch_size, :] += patches[idx].reshape(patch_size, patch_size, 3)
            count[i:i+patch_size, j:j+patch_size, :] += 1
            idx += 1
    return image / count
