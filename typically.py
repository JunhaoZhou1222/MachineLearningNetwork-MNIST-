from euclidean import euclidean_distance
from k_mean_distance import compute_k_mean_distance

def compute_typically(feature_vector, k, eps=1e-6):
    typically = []
    mean_dis = compute_k_mean_distance(feature_vector, k)
    for i in range(len(feature_vector)):
        typically_result = 1 / (mean_dis[i] + eps)
        typically.append(typically_result)
    return typically

if __name__ == "__main__":
    feature_vector = [[0, 1], [9, 6], [0, 1], [0, 1]]
    k = 3
    result = compute_typically(feature_vector, k, eps=1e-6)
    print(result)
