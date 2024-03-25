import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
from sklearn.metrics import pairwise_distances 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def metric_cosine(adv1, adv2):
    cosine = matrix_cosine(adv1, adv2)
    return cosine


def metric_aggressiveness(original_x, adv_x, original_y,model,metric="euclidean" , ord=2, dataset="neris", adv_label=1):
    predicted_x = model.predict(original_x)
    neg_label = 1 - original_y
    idx_adv = np.squeeze(np.argwhere((original_y == adv_label) & (predicted_x == adv_label)))
    idx = np.squeeze(np.argwhere((original_y == neg_label) & (predicted_x == neg_label)))
    initial_adv = original_x[idx_adv]
    negative_data = original_x[idx]

    nn_dist = get_distance(initial_adv, negative_data, metric=metric, ord=ord)
    pert_size = calculate_pert_size(initial_adv, adv_x)
    aggressiveness = pert_size / nn_dist

    return aggressiveness


def calculate_nn_dist(negative_examples, positive_examples):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric="chebyshev").fit(negative_examples)
    distances, indices = nbrs.kneighbors(positive_examples)
    return distances

def calculate_pert_size(original, perturbed, ord=np.inf):
    if ord=="cosine":
        dist = 1- matrix_cosine(original, perturbed)
    else:
        dist = np.linalg.norm(original - perturbed, ord=ord, axis=1)
    return dist 


def get_distance(initial, negative_data, metric, ord):
    nn_dist = pairwise_distances(initial, negative_data, metric=metric, n_jobs=10 )
    if metric=="cosine":
        idx_euc = np.argmax(nn_dist, axis=1)
    else:
        idx_euc = np.argmin(nn_dist, axis=1)

    negative_data = negative_data[idx_euc] 
    if ord=="cosine":
        dist = 1 - matrix_cosine(initial, negative_data)
    else:
        dist = np.linalg.norm(negative_data - initial, ord=ord, axis=1)

    return np.squeeze(dist)


def matrix_cosine(x, y):
    return np.einsum('ij,ij->i', x, y) / (
              np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))


def plot_feature_perturbation(original, adversarials, fig_path):
    diff = np.absolute(original-adversarials)
    pert_feat_count = np.count_nonzero(diff, axis=0)
    fig = plt.figure()
    plt.rcParams.update({'font.size': 20})
    index = np.arange(0, pert_feat_count.shape[0])
    plt.scatter(index, pert_feat_count)
    plt.ylabel("Count")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.savefig(fig_path)


def visualize_embeddings(data , labels, label_names, color, graph_path, rs, tsne_comp, pca_comp, perplexity):
    dim_reducer = TSNE(n_components=tsne_comp, random_state=rs, perplexity=perplexity)
    pca_30 = PCA(n_components=pca_comp, random_state=rs)
    pca_result_30 = pca_30.fit_transform(data)
    print("PCA completed")
    tsne_results = dim_reducer.fit_transform(pca_result_30)
    df = pd.DataFrame.from_dict({'x':tsne_results[:,0], 'y':tsne_results[:,1], 'label':labels})
    fig, ax = plt.subplots()
    palette = sns.color_palette("bright", color)
    sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax, palette=palette, alpha=0.4)
    ax.legend(handles=ax.legend_.legendHandles, labels=label_names, loc="lower left")
    plt.tight_layout()
    plt.savefig(graph_path)