import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy

def calculate_statistics(image_path):
   
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("Error loading image. Please check the file path.")
    
    normalized_img = image / 255.0

    pixel_values = normalized_img.flatten()

    variance = np.var(pixel_values)
    
    image_skewness = skew(pixel_values)
    
    image_kurtosis = kurtosis(pixel_values)
    
    image_entropy = shannon_entropy(image)

    return {
        'variance': variance,
        'skewness': image_skewness,
        'kurtosis': image_kurtosis,
        'entropy': image_entropy
    }

image_path = 'C:\\Users\\guruc\\OneDrive\\Desktop\\Project\\FakeNote.jpg' 
stats = calculate_statistics(image_path)

print(f"Variance: {stats['variance']}")
print(f"Skewness: {stats['skewness']}")
print(f"Kurtosis: {stats['kurtosis']}")
print(f"Entropy: {stats['entropy']}")
