import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from features.color_features import extract_color_features
from features.texture_features import extract_texture_features
from utils.preprocessing import preprocess_image
import numpy as np

def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    labels = df.drop('image_id', axis=1).idxmax(axis=1).values
    image_paths = [os.path.join(image_dir, f"{img_id}.jpg") for img_id in df['image_id']]

    X = []
    y = []

    for path, label in tqdm(zip(image_paths, labels), total=len(labels)):
        image, gray = preprocess_image(path)
        color_feat = extract_color_features(image)
        texture_feat = extract_texture_features(gray)
        features = np.hstack([color_feat, texture_feat])
        X.append(features)
        y.append(label)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return np.array(X), np.array(y), le

