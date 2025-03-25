# Logo-Similarity

# Logo Similarity Clustering

## Overview
This project aims to cluster logos based on visual similarities using deep learning and K-Means clustering. Given a dataset containing domain names, the script automatically fetches logos from Clearbit, extracts features using a pre-trained VGG16 model, and clusters them based on visual characteristics.

## Approach

### 1. **Fetching Logos**
- The script reads a dataset (`.parquet` or `.csv`) containing domain names.
- It constructs logo URLs using Clearbit's logo API (`https://logo.clearbit.com/{domain}`).
- Logos are downloaded concurrently using `ThreadPoolExecutor` for efficiency.
- The script tracks successful and failed downloads and calculates the success rate.

### 2. **Preprocessing Logos**
- The downloaded logos are resized to `224x224` pixels to match the input format of VGG16.
- Images are converted to RGB format and normalized using `preprocess_input` from TensorFlow.

### 3. **Feature Extraction Using VGG16**
- A pre-trained VGG16 model (without the top classification layer) is used to extract deep features.
- Each logo is passed through the model, and the extracted features are flattened into a vector.

### 4. **Clustering Using K-Means**
- The extracted features are clustered using K-Means with different numbers of clusters (`[3, 5, 10, 20, 50, 70, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]`).
- The optimal number of clusters can be determined by analyzing the results.
- Logos are organized into directories based on their assigned cluster labels.

### 5. **Observations**
- After testing with multiple cluster values (`[3, 5, 10, 20, 50, 70, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]`), I observed that **at 5 clusters**, visually similar logos (e.g., **Toyota, Hyundai, and Honda**) were grouped together more effectively compared to other logos. This indicates that the clustering approach successfully captures visual similarities between logos.

### 6. **Saving and Visualizing Results**
- Logos are saved into corresponding cluster folders.
- The script generates plots of clusters for visual analysis.

## Requirements
To run this project, install the required dependencies:
```bash
pip install numpy pandas requests opencv-python tensorflow scikit-learn matplotlib
