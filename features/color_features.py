import numpy as np

def extract_color_features(image):
  mean=np.mean(image,axis=(0,1));
  std=np.std(image,axis=(0,1));
  return np.concatenate([mean,std])
