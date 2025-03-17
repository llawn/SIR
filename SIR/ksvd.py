import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from PIL import Image
from scipy.linalg import svd
import matplotlib.pyplot as plt


def load_image(image_path, crop_shape=(100, 100)):
    """
    Load an image and crop a region of the specified shape from the center.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    
    # Calculate the starting coordinates for the center crop
    height, width, _ = image.shape
    start_i = (height - crop_shape[0]) // 2
    start_j = (width - crop_shape[1]) // 2
    
    # Crop the center region
    image = image[start_i:start_i+crop_shape[0], start_j:start_j+crop_shape[1], :]
    return image
    image = image[:crop_shape[0], :crop_shape[1], :]
    return image


def extract_patches(image: np.array, patch_size: int) -> np.array:
    patches = []
    for i in range(image.shape[0] - patch_size + 1):
        for j in range(image.shape[1] - patch_size + 1):
            patch = image[i:i+patch_size, j:j+patch_size, :].flatten()
            patches.append(patch)
    return np.array(patches)


# Load images
image = load_image('image.jpg')
noisy_image = load_image('noisy_image.jpg')


def save_image(image, output_path):
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(output_path)


save_image(image, "image_crop.jpg")
save_image(noisy_image, "noisy_image_crop.jpg")

# Extract patches
patch_size = 8
patches = extract_patches(noisy_image, patch_size)

# Verify the shape of patches
print("Shape of patches:", patches.shape)  # Should be (num_patches, patch_size * patch_size * 3)


def initialize_dictionary(patch_size, num_atoms):
    return np.random.randn(patch_size * patch_size * 3, num_atoms)


num_atoms = 256
D = initialize_dictionary(patch_size, num_atoms)


def modified_omp(D, patches, gamma):
    num_patches = patches.shape[0]
    num_atoms = D.shape[1]
    alpha = np.zeros((num_atoms, num_patches))  # Store coefficients for all patches
    
    # Construct the matrix K
    K = np.kron(np.eye(3), np.ones((num_atoms // 3, num_atoms // 3))) / (num_atoms // 3)
    
    # If K is smaller than np.eye(num_atoms), pad it with zeros
    if K.shape[0] < num_atoms:
        pad_size = num_atoms - K.shape[0]
        K = np.pad(K, ((0, pad_size), (0, pad_size)), mode='constant')
    
    # Modify the dictionary D
    D_tilde = D @ (np.eye(num_atoms) + gamma * K)
    
    # Compute sparse coefficients for each patch
    for i in range(num_patches):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
        omp.fit(D_tilde, patches[i])  # patches[i] should be a 1D array
        alpha[:, i] = omp.coef_
    
    return alpha


gamma = 5.25
alpha = modified_omp(D, patches, gamma)  # alpha is now (num_atoms, num_patches)


def ksvd_update(D, patches, alpha, gamma, num_iterations=10):
    num_atoms = D.shape[1]
    
    for _ in range(num_iterations):
        for i in range(num_atoms):
            # Find patches that use the i-th atom
            indices = np.where(alpha[i, :] != 0)[0]
            if len(indices) == 0:
                continue
            
            # Compute the error matrix E
            E = patches[indices].T - D @ alpha[:, indices]  # Transpose patches[indices]
            
            # Update the i-th atom and its corresponding coefficients
            U, S, Vt = svd(E, full_matrices=False)
            D[:, i] = U[:, 0]
            alpha[i, indices] = S[0] * Vt[0, :]
    
    return D, alpha


D, alpha = ksvd_update(D, patches, alpha, gamma)

def reconstruct_image(patches, D, alpha, image_shape, patch_size):
    height, width, _ = image_shape
    reconstructed_image = np.zeros(image_shape)
    count = np.zeros(image_shape)
    idx = 0
    
    for i in range(0, height - patch_size + 1):
        for j in range(0, width - patch_size + 1):
            patch = D @ alpha[:, idx]
            patch = patch.reshape(patch_size, patch_size, 3)
            reconstructed_image[i:i+patch_size, j:j+patch_size, :] += patch
            count[i:i+patch_size, j:j+patch_size, :] += 1
            idx += 1
    
    reconstructed_image /= count
    return reconstructed_image

reconstructed_image = reconstruct_image(patches, D, alpha, image.shape, patch_size)


save_image(reconstructed_image, 'denoised_image.jpg')
