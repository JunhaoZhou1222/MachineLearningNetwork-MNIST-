from euclidean import euclidean_distance

def compute_k_mean_distance(feature_vector, k):
    result = []
    for i, v1 in enumerate(feature_vector):
        distance = []
        for j, v2 in enumerate(feature_vector):
            if i != j:
                distance.append(euclidean_distance(v1[0],v2[0],v1[1],v2[1]))

        sorted_distance = sorted(distance)
        print(sorted_distance)

        total_distance = 0
        for x in range(k):
            total_distance += sorted_distance[x]
        total_distance /= k
        result.append(total_distance)
    return result
    
if __name__ == "__main__":
    feature_vector = [[0, 1], [9, 6], [0, 1], [0, 1]]
    k = 3
    final_result = compute_k_mean_distance(feature_vector, k)
    print(final_result)

