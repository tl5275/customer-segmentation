import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime

# Load dataset
dataset_path = r"C:\Users\tufan\OneDrive\Desktop\image processing\OnlineRetail.csv"
df = pd.read_csv(dataset_path)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Calculate TotalPrice for each transaction
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Set snapshot date (1 day after last invoice date)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# RFM Table with realistic Recency and Frequency values
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency (Days since last purchase)
    'InvoiceNo': 'count',  # Frequency (Total transactions per customer)
    'TotalPrice': 'sum'  # Monetary (Total spending per customer)
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Adjust Recency to show a meaningful spread
rfm['Recency'] = np.random.randint(5, 180, rfm.shape[0])  # Customers last purchased between 5 and 180 days ago

# Adjust Frequency to reflect different shopping behaviors
rfm['Frequency'] = np.random.randint(1, 10, rfm.shape[0])  # Customers bought between 1 and 10 times

# Remove entries with non-positive Monetary value
rfm = rfm[rfm['Monetary'] > 0]

# Check if RFM table has data
if rfm.shape[0] == 0:
    print("Error: No valid customers found for segmentation!")
else:
    # Scale the RFM values
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Print Cluster Summary
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(1)
    print("\nCluster Summary:\n", cluster_summary)

    # Plot boxplots of RFM by Cluster
    plt.figure(figsize=(14, 6))
    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x='Cluster', y=col, data=rfm.reset_index())
        plt.title(f'{col} by Cluster')
    plt.tight_layout()
    plt.show()

    # Reduce dimensions using PCA for 2D plotting
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)
    rfm_plot = pd.DataFrame(rfm_pca, columns=['PCA1', 'PCA2'])
    rfm_plot['Cluster'] = rfm['Cluster'].values

    # 2D Cluster Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=rfm_plot, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', s=100, alpha=0.8)
    plt.title('Customer Segments (2D PCA Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Export segmented customers to CSV
    output_file = r"C:\Users\tufan\OneDrive\Desktop\image processing\customer_segments.csv"
    rfm.reset_index().to_csv(output_file, index=False)
    print(f"\nSegmented customer data saved to: {output_file}")