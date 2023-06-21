import math
import operator
import sys

import numpy as np
from PIL import Image


class KMeansJava:

    @staticmethod
    def euclidean_distance(point1, point2):
        return math.dist(point1, point2)

    @staticmethod
    def random_centroids(rgb, k, seed=None):
        if seed is not None:
            np.random.seed(seed)
        length = len(rgb)
        random_centroids = []
        for i in range(k):
            rand_idx = int(np.random.rand() * length)
            random_centroids.append(rgb[rand_idx])
        return random_centroids

    @staticmethod
    def k_means_helper(img, k, seed=None):
        w = img.width
        h = img.height
        #  Read rgb values from the image
        rgb = [0] * (w * h)
        count = 0
        for i in range(w):
            for j in range(h):
                rgb[count] = img.getpixel((i, j))
                count += 1
        #  Call k means algorithm: update the rgb values
        KMeansJava.k_means(rgb, k, seed)
        #  Write the new rgb values to the image
        count = 0
        for i in range(w):
            for j in range(h):
                img.putpixel((i, j), rgb[count])
                count += 1
        return img

    # Your k-means code goes here
    # Update the list rgb by assigning each entry in the rgb list to its cluster center
    @staticmethod
    def k_means(rgb, k, seed):
        length = len(rgb)
        converged = False
        iterations = 10
        cluster_centers_rgb = KMeansJava.random_centroids(rgb, k, seed)
        cluster_of_instance = [0] * length
        while (iterations != 0) & (not converged):
            for idx in range(length):
                min_dist = sys.maxsize
                min_k = -1
                for j in range(k):
                    distance = KMeansJava.euclidean_distance(cluster_centers_rgb[j], rgb[idx])
                    if distance < min_dist:
                        min_dist = distance
                        min_k = j
                cluster_of_instance[idx] = min_k
            new_cluster_centers_rgb = []
            for j in range(k):
                points_sum = (0, 0, 0)
                c = 0
                for idx in range(length):
                    if cluster_of_instance[idx] == j:
                        points_sum = tuple(map(operator.add, points_sum, rgb[idx]))
                        c += 1
                if c == 0:
                    c = 1
                # tuple(map(lambda x: int(x / y), numbers_tuple))
                new_cluster_centers_rgb.append(tuple(int(ps / c) for ps in points_sum))
            iterations -= 1
            if new_cluster_centers_rgb == cluster_centers_rgb:  # Converged if previous equals current.
                converged = True
        for idx in range(length):
            rgb[idx] = cluster_centers_rgb[cluster_of_instance[idx]]


def main(arg_list):
    if len(arg_list) < 3:
        print("Usage: KMeans <input-image> <k> <output-image>")
        return

    seed = 131
    img_path = arg_list[1]
    k = int(arg_list[2])
    save_path = arg_list[3]

    k_means_jpg = KMeansJava.k_means_helper(Image.open(img_path), k, seed)
    k_means_jpg.save(save_path)
    k_means_jpg.show()


if __name__ == "__main__":
    main(sys.argv)
