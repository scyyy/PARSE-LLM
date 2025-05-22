import re
import time
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.utils import gen_batches
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

import numpy as np
from itertools import combinations
from joblib import Parallel, delayed

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer

from tqdm import tqdm


def TFIDF(log_content):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(log_content).toarray()

    return X


def SentenceBERT(log_content):
    model = SentenceTransformer('../models/all-MiniLM-L12-v2')
    X = model.encode(log_content, show_progress_bar=False)

    return X


def lcs(log_content):
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    n_logs = len(log_content)
    similarity_matrix = np.zeros((n_logs, n_logs))

    def compute_lcs(i, j):
        return lcs_length(log_content[i], log_content[j])

    pairs = list(combinations(range(n_logs), 2))
    results = Parallel(n_jobs=-1)(delayed(compute_lcs)(i, j) for i, j in pairs)
    for idx, (i, j) in enumerate(pairs):
        similarity_matrix[i, j] = similarity_matrix[j, i] = results[idx]

    max_lcs = np.max(similarity_matrix)
    distance_matrix = max_lcs - similarity_matrix

    return distance_matrix


def extract_special_chars(input_string):
    special_chars = re.findall(r'[^\w\s]', input_string)
    return special_chars


def special_chars_clustering(log_cluster_result):
    fine_grained_clusters = {}

    for key, logs in log_cluster_result.items():
        fine_grained_cluster = {}
        cluster_start_num = 0

        for log in logs:
            special_char = extract_special_chars(log)
            special_char = str(special_char)

            if special_char not in fine_grained_cluster:
                fine_grained_cluster[special_char] = [log]
            else:
                fine_grained_cluster[special_char].append(log)

        new_fine_grained_clusters = {}

        for i, (special_char, logs) in enumerate(fine_grained_cluster.items()):
            new_key = f"{key}-{i + 1}" 
            new_fine_grained_clusters[new_key] = logs

        fine_grained_clusters.update(new_fine_grained_clusters)

    return fine_grained_clusters


def GMM_clustering(log_content, similarity_function):
    start_time = time.time()

    if similarity_function == "tfidf":
        X = TFIDF(log_content)
    elif similarity_function == "bert":
        X = SentenceBERT(log_content)
    elif similarity_function == "lcs":
        X = lcs(log_content)
    else:
        raise ValueError("Unrecognized similarity function.")

    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X)

    max_components = 150

    def compute_silhouette_score(n):
        gmm = GaussianMixture(n_components=n, covariance_type='full', n_init=10)
        labels = gmm.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, labels)
        return score

    silhouette_scores = Parallel(n_jobs=-1)(delayed(compute_silhouette_score)(n) for n in tqdm(range(2, max_components + 1), desc='calculating silhouette_score'))

    best_n_components = np.argmax(silhouette_scores) + 2 

    gmm = GaussianMixture(n_components=best_n_components, covariance_type='full', n_init=50)
    gmm.fit(X_reduced)

    labels = gmm.predict(X_reduced)

    labeled_logs = [[log_content[i], str(labels[i])] for i in range(len(log_content))]

    log_cluster_result = {}
    for log, label in labeled_logs:
        if label not in log_cluster_result:
            log_cluster_result[label] = [log]
        else:
            log_cluster_result[label].append(log)

    fine_grained_clusters = special_chars_clustering(log_cluster_result)

    final_labeled_logs = []

    for log in log_content:
        for cluster_label, logs in fine_grained_clusters.items():
            if log in logs:
                final_labeled_logs.append([log, cluster_label])
                break

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The function executed in {execution_time:.6f} seconds")

    return final_labeled_logs

from tqdm import tqdm

def GMM_clustering_with_dynamic_components(log_content, similarity_function, change_rate_threshold=0.02, min_components=2, max_iter=1000):

    start_time = time.time()

    if similarity_function == "tfidf":
        X = TFIDF(log_content)
    elif similarity_function == "bert":
        X = SentenceBERT(log_content)
    elif similarity_function == "lcs":
        X = lcs(log_content)
    else:
        raise ValueError("Unrecognized similarity function.")

    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X)

    prev_score = -1
    silhouette_scores = []
    best_n_components = None

    for n in tqdm(range(min_components, max_iter + 1), desc="Calculating silhouette scores"):
        gmm = GaussianMixture(n_components=n, covariance_type="full", n_init=10)
        labels = gmm.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, labels)
        silhouette_scores.append(score)

        if prev_score != -1:
            change_rate = abs(score - prev_score) / max(abs(prev_score), 1e-10)
            if change_rate < change_rate_threshold:
                best_n_components = n
                print(f"Optimal number of clusters determined at {n} (change rate: {change_rate:.4f})")
                break
        prev_score = score

    if best_n_components is None:
        best_n_components = np.argmax(silhouette_scores) + min_components
        print(f"Optimal number of clusters not found dynamically; using maximum silhouette score at {best_n_components} clusters.")

    gmm = GaussianMixture(n_components=best_n_components, covariance_type="full", n_init=50)
    gmm.fit(X_reduced)
    labels = gmm.predict(X_reduced)

    labeled_logs = [[log_content[i], str(labels[i])] for i in range(len(log_content))]

    log_cluster_result = {}
    for log, label in labeled_logs:
        if label not in log_cluster_result:
            log_cluster_result[label] = [log]
        else:
            log_cluster_result[label].append(log)

    fine_grained_clusters = special_chars_clustering(log_cluster_result)

    final_labeled_logs = []

    for log in log_content:
        for cluster_label, logs in fine_grained_clusters.items():
            if log in logs:
                final_labeled_logs.append([log, cluster_label])
                break

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"The function executed in {execution_time:.6f} seconds")

    return final_labeled_logs


def HDBSCAN(log):
    def clean_log(log):
        log = re.sub(r'ContainerId: \w+', '[ContainerId]', log)
        log = re.sub(r'NodeId: [\w\.:-]+', '[NodeId]', log)
        log = re.sub(r'Token: \{.*?\}', '[Token]', log)
        log = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP]', log)
        return log

    for ids in tqdm(grouped_by_length.keys()):
        target_log = grouped_by_length[ids]
        logs = [clean_log(i[0]) for i in target_log] 
        model = SentenceTransformer('../models/all-L6-v2')
        embeddings = model.encode(logs)

        clusterer = hdbscan.HDBSCAN(
            gen_min_span_tree=True,  
            min_cluster_size=2,      
            min_samples=1,
            cluster_selection_epsilon=0.5,  
            cluster_selection_method='eom'  

        )
        clusters = clusterer.fit_predict(embeddings)