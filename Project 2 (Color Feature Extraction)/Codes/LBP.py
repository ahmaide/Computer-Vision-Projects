from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics import auc
from skimage import feature

def load_images_from_folder(folder):
    images_CV = []
    images_plot = []
    for i in range(1000):
        filename = os.path.join(folder, f"{i}.jpg")
        try:
            img = cv2.imread(filename)
            images_CV.append(img)
            img = Image.open(filename)
            images_plot.append(img)
        except IOError:
            print(f"Error opening {filename}")
            continue
    return images_CV, images_plot


def extract_lbp_features(image, numPoints=24, radius=8, eps=1e-7):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(image, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist


def calculate_distance(features1, features2):
    return np.linalg.norm(features1 - features2)


def retrieve_images(query_hist, all_histograms, threshold):
    distances = [(index, calculate_distance(query_hist, hist)) for index, hist in enumerate(all_histograms)]
    sorted_distances = sorted((index_distance for index_distance in distances), key=lambda x: x[1])[1:]
    filtered_distances = [element for element in sorted_distances if element[1] <= threshold]
    return [index for index, distance in filtered_distances]


def compute_metrics(retrieved_indices, query_label):
    relevant_indices = set(range(query_label*100, (query_label+1)*100))
    retrieved_relevant = relevant_indices.intersection(set(retrieved_indices))
    precision = len(retrieved_relevant) / len(retrieved_indices) if len(retrieved_indices) > 0 else 0
    recall = len(retrieved_relevant) / 99
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1_score


def plot_all_queries(results, dataset):
    num_columns = 6
    num_rows = 2
    for i in range(0, len(results), num_rows):
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(3 * num_columns, 3 * num_rows))

        for row in range(num_rows):
            if i + row < len(results):
                result = results[i + row]
                query_index = result['query_index']
                retrieved_indices = result['retrieved']
                query_img = dataset[query_index]
                axs[row, 0].imshow(query_img)
                axs[row, 0].set_title(f"Query: {query_index}")
                axs[row, 0].axis('off')
                max_similar = min(len(retrieved_indices), num_columns - 1)
                for j in range(1, num_columns):
                    if j <= max_similar:
                        idx = retrieved_indices[j - 1]
                        similar_img = dataset[idx]
                        axs[row, j].imshow(similar_img)
                        axs[row, j].set_title(f"Similar {j}: {idx}")
                        axs[row, j].axis('off')
                    else:
                        axs[row, j].axis('off')
            else:
                for col in range(num_columns):
                    axs[row, col].axis('off')

        plt.suptitle(f"CBIR Results for LBP", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def calculate_tpr_fpr(query_index, retrieved_indices):
    query_label = query_index // 100
    true_positives = len([i for i in retrieved_indices if i // 100 == query_label])
    false_positives = len(retrieved_indices) - true_positives
    tpr = true_positives / 99
    fpr = false_positives / 900
    return tpr, fpr


def plot_ROC(roc_auc,fprs, tprs):
    plt.figure()
    plt.plot(fprs, tprs, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for LBP')
    plt.legend(loc="lower right")
    plt.show()


def find_max_threshold(query_images, histograms):
    max_threshold = max([max([calculate_distance(histogram, histograms[query_image])
                     for histogram in histograms]) for query_image in query_images])
    return max_threshold


def print_results(results, threshold):
    average_precision = np.mean([r['precision'] for r in results])
    average_recall = np.mean([r['recall'] for r in results])
    average_f1_score = np.mean([r['f1_score'] for r in results])
    average_time = np.mean([r['time'] for r in results])
    retrived = np.mean([r['Images'] for r in results])
    print(f"For the threshold {threshold}:")
    print(f"\tAverage Precision: {average_precision}")
    print(f"\tAverage Recall: {average_recall}")
    print(f"\tAverage F1 Score: {average_f1_score}")
    print(f"\tAverage Time: {average_time} seconds")
    print(f"\tAverage Number of Retrieved Images: {retrived} image")



folder = "dataset"
dataset_CV, dataset_plot = load_images_from_folder(folder)
query_indexes = [13, 134, 212, 320, 496, 500, 628, 738, 879, 941]

tprs = []
fprs = []
all_histograms = [extract_lbp_features(img) for img in dataset_CV]
max_threshold = find_max_threshold(query_indexes, all_histograms)
thresholds = np.linspace(0, max_threshold, 100)
for i, threshold in enumerate(thresholds):
    tpr_avg = 0
    fpr_avg = 0
    results = []
    for query_label, query_index in enumerate(query_indexes):
        query_hist = all_histograms[query_index]
        start_time = time.time()
        retrieved_indices = retrieve_images(query_hist, all_histograms, threshold)
        precision, recall, f1_score = compute_metrics(retrieved_indices, query_label)
        tpr, fpr = calculate_tpr_fpr(query_index, retrieved_indices)
        end_time = time.time()
        execution_time = end_time - start_time
        results.append({
            "query_index": query_index, "precision": precision,
            "recall": recall, "f1_score": f1_score,
            "time": execution_time, "retrieved": retrieved_indices[:5],
            "Images": len(retrieved_indices)
        })
        tpr_avg += tpr
        fpr_avg += fpr
    if i % 20 == 19:
        print_results(results, threshold)
    tprs.append(tpr_avg / 10)
    fprs.append(fpr_avg / 10)
    if i == 99:
        plot_all_queries(results, dataset_plot)

roc_auc = auc(fprs, tprs)
print(f"::::::::::::::::::::::: AUC: {roc_auc} :::::::::::::::::::::::")
plot_ROC(roc_auc, fprs, tprs)