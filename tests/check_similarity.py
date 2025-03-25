import numpy as np
import pandas as pd
import os
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from glob import glob

def get_logo(domain):
    return f"https://logo.clearbit.com/{domain}"

def download_logo(domain):
    logo_url = get_logo(domain)
    response = requests.get(logo_url)

    if response.status_code == 200:
        file_path = f"logos/{domain}.png"
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"✅ Downloaded: {domain}")
    else:
        print(f"❌ Logo not found: {domain}")

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = preprocess_input(img)
    return img

def extract_features(image_paths, model):
    features = []
    for img_path in image_paths:
        img = load_and_preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img)
        feature = feature.flatten()
        features.append(feature)
    return np.array(features)

def main(input_file, num_clusters):
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Input file must be a CSV or Parquet file")
    
    if "domain" not in df.columns:
        raise ValueError("Input file must contain a 'domain' column")
    
    os.makedirs("logos", exist_ok=True)
    
    domains = df["domain"].unique()
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_logo, domains)
    
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    image_paths = glob("logos/*.png")
    features = extract_features(image_paths, model)
    
    print(f"Running K-Means with {num_clusters} clusters")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    labels = kmeans.labels_
    
    os.makedirs("clusters", exist_ok=True)
    
    for cluster in range(num_clusters):
        cluster_dir = os.path.join("clusters", f"cluster_{cluster}")
        os.makedirs(cluster_dir, exist_ok=True)
        
    for img_path, label in zip(image_paths, labels):
        img_name = os.path.basename(img_path)
        dest_path = os.path.join("clusters", f"cluster_{label}", img_name)
        cv2.imwrite(dest_path, cv2.imread(img_path))
    
    print("Clustering complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster logos using K-Means")
    parser.add_argument("input_file", type=str, help="Path to the input CSV or Parquet file")
    parser.add_argument("num_clusters", type=int, help="Number of clusters for K-Means")
    args = parser.parse_args()
    
    main(args.input_file, args.num_clusters)
