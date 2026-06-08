from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def train_model(df_scaled):
    print("K-Means model will be trained to segment customer data into 4 clusters according to their purchasing behaviour")
    # Initialize K-Means model
    kmeans = KMeans(
        n_clusters=4,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )

    # Train model and assign cluster labels
    cluster_labels = kmeans.fit_predict(df_scaled)

    # Add cluster labels to original RFM table
    df_scaled['Cluster'] = cluster_labels
    print("RFM dataframe updated, now includes the cluster label for each customer, " \
    "indicating the group they belong \n")

    return df_scaled

def plot_2d_segmentation(df_cluster):
    # Color palette for cluster differenciation
    qual_palette = sns.color_palette("bright", 4)

    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    sns.scatterplot(data=df_cluster, x='Frequency', y='Monetary', hue='Cluster', palette=qual_palette, ax=axes[0])
    axes[0].set_title("Customer Segments by Frequency and Monetary")
    sns.scatterplot(data=df_cluster, x='Frequency', y='Recency', hue='Cluster', palette=qual_palette, ax=axes[1])
    axes[1].set_title("Customer Segments by Frequency and Recency")
    sns.scatterplot(data=df_cluster, x='Recency', y='Monetary', hue='Cluster', palette=qual_palette, ax=axes[2])
    axes[2].set_title("Customer Segments by Recency and Monetary")

    fig.suptitle("Customer Segments after K_Means implementation", fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig("../figures/k-means_segments_3_angles.png")
    plt.show()

def plot_3d_segmentation(df_cluster):
    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(111, projection='3d')

    # Scatterplot
    scatter = ax.scatter(
        df_cluster['Recency'],
        df_cluster['Frequency'],
        df_cluster['Monetary'],
        c=df_cluster['Cluster'],
        cmap='viridis',
        alpha=0.5
    )

    # Labels
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')

    plt.title("3D Customer Segmentation", fontsize=18, fontweight='bold')

    # Legend
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Clusters"
    )

    ax.add_artist(legend)

    plt.savefig("../figures/k-means_segments_3d.png")
    plt.show()

def main(path):
    print(f"Path: {path} \n") 

    df_rfm_scaled = load_data(path)
    print("The scaled rfm dataset has been loaded \n")

    df_rfm_cluster = train_model(df_rfm_scaled)
    print("The k-means model has been trained \n")

    df_rfm_cluster.to_csv("../data_clean/rfm_cluster.csv")
    print("Updated dataframe has been saved in '/data_clean/rfm_cluster.csv' \n")

    plot_2d_segmentation(df_rfm_cluster)
    print("2D plot of the segmentation was generated \n")

    plot_3d_segmentation(df_rfm_cluster)
    print("3D plot of the segmentation was generated \n")

if __name__ == "__main__":
    rfm_scaled_path = "../data_clean/rfm_scaled.csv"
    main(rfm_scaled_path)