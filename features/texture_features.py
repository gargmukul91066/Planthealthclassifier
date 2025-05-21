import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

def extract_texture_features(image_gray):
    lbp = local_binary_pattern(image_gray, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    glcm = greycomatrix(image_gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    return np.hstack([hist, contrast, dissimilarity, homogeneity, energy, correlation])
