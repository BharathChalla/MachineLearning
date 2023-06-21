import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue

import numpy as np
import pandas as pd
from PIL import Image


class KMeans:
    def __init__(self, img_path, k=5, seed=None):
        self.img = Image.open(img_path)
        self.img_copy = self.img.copy()
        self.pxl_map = self.img.load()
        self.k = k
        self.cluster_map = np.ones((self.img.height, self.img.width))
        self.centroids = self.random_centroids(self.k, self.img, seed)
        self.converged = False

    @staticmethod
    def random_centroids(k, img, seed=None):
        if seed is not None:
            np.random.seed(seed)
        random_centroids = []
        for i in range(k):
            rand_x = np.random.rand() * img.width
            rand_y = np.random.rand() * img.height
            random_centroids.append((rand_x, rand_y))
        return random_centroids

    @staticmethod
    def euclidean_distance(point1, point2):
        sq_sum = 0
        for a, b in zip(point1, point2):
            sq_sum += ((a - b) ** 2)
        return math.sqrt(sq_sum)

    def generate_cluster_map(self):
        cluster_map = self.cluster_map
        for x in range(self.img.width):
            for y in range(self.img.height):
                dist = []
                for i in range(self.k):
                    cluster_pixel = self.img_copy.getpixel(self.centroids[i])
                    point_pixel = self.img_copy.getpixel((x, y))
                    euclidean_dist = self.euclidean_distance(cluster_pixel, point_pixel)
                    dist.append(euclidean_dist)
                index = dist.index(min(dist)) + 1
                cluster_map[y][x] = index
                self.pxl_map[x, y] = self.img_copy.getpixel(self.centroids[index - 1])
        if (cluster_map == self.cluster_map).all():  # Converged if previous equals current.
            self.converged = True
        return cluster_map

    # Your k-means code goes here
    # Update the list rgb by assigning each entry in the rgb list to its cluster center
    def k_means(self):
        cl_map = self.generate_cluster_map()
        new_centroids = self.centroids.copy()  # To retain the old Centroid if n == 0
        for i in range(self.k):
            x_values = y_values = n = 0
            for x in range(self.img.height):
                for y in range(self.img.width):
                    if cl_map[x, y] == i:
                        x_values += x
                        y_values += y
                        n += 1
            if n > 0:
                new_centroids[i] = (x_values / n, y_values / n)
        self.centroids = new_centroids

    def cluster(self):
        while not self.converged:
            self.k_means()
            self.cluster_map = self.generate_cluster_map()

    def get_compressed_image(self):
        return self.img


params_k_means = {
    'img_paths': ["data/kmeans/Koala.jpg", "data/kmeans/Penguins.jpg"],
    'comp_img_save_paths': ["data/kmeans/Koala", "data/kmeans/Penguins"],
    'k_values': [2, 5, 10, 15, 20],
    'seeds': [8191, 131071, 524287, 6700417, 2147483647],  # 131, 151, 1531
}


def get_param_values(params):
    k_values = params['k_values']
    seeds = params['seeds']
    for k in k_values:
        for seed in seeds:
            yield k, seed


def get_k_means_compressed_img_multi(q, img_path, k, seed):
    k_means_instance = KMeans(img_path, k, seed)
    k_means_instance.cluster()
    q.put((img_path, k, seed, k_means_instance.get_compressed_image()))


def get_k_means_compressed_img(img_path, k, seed):
    k_means_instance = KMeans(img_path, k, seed)
    k_means_instance.cluster()
    return k_means_instance.get_compressed_image()


def get_img_size(img_path):
    return os.stat(img_path).st_size / 1024


def get_column_names(num_cols, values):
    rng = range(1, num_cols + 1)
    new_cols = ["seed{}({})".format(i, v) for i, v in zip(rng, values)]
    return new_cols[:num_cols]


def main():
    arg_list = sys.argv
    if len(arg_list) != 4:
        print("Usage: KMeans <input-image> <k> <output-image>")

        print("Running on all parameters")
        orig_img_paths = params_k_means['img_paths']
        comp_img_save_paths = params_k_means['comp_img_save_paths']
        k_values = params_k_means['k_values']
        seeds = params_k_means['seeds']
        values_table = []
        for orig_img_path, save_path in zip(orig_img_paths, comp_img_save_paths):
            original_img_size = get_img_size(orig_img_path)
            print("Original Image Size: ", original_img_size, "KB")
            K = set()
            compression_ratios = []
            run_times = []
            values_table.append((None, None, original_img_size, 1, None))
            q = Queue()
            with ThreadPoolExecutor(max_workers=8) as executor:
                arguments = []
                for k, seed in get_param_values(params_k_means):
                    arguments.append((orig_img_path, k, seed))
                    if len(arguments) == 5:
                        start_time = time.time()
                        result = executor.map(lambda p: get_k_means_compressed_img(*p), arguments)

                        for k_means_jpg in result:
                            K.add(k)
                            print("\nValues - ", "K: ", str(k), ", Seed: ", str(seed), sep="")
                            comp_img_path = save_path + '-K' + str(k).zfill(2) + '-S' + str(seed).zfill(
                                10) + 'multi.jpg'
                            k_means_jpg.save(comp_img_path)  # format='JPG'
                            compressed_img_size = get_img_size(comp_img_path)
                            # https://en.wikipedia.org/wiki/Data_compression_ratio
                            compression_ratio = original_img_size / compressed_img_size
                            compression_ratios.append(compression_ratio)
                            run_time = round(time.time() - start_time, 3)
                            run_times.append(run_time)
                            print("Compressed image Size:", round(compressed_img_size, 3), "KB ", "Run time:", run_time,
                                  "s")
                            values_table.append((k, seed, compressed_img_size, compression_ratio, run_time))
                        q = Queue()
                        print("Cleared Queue")

            K = list(K)
            compression_ratios = np.array(compression_ratios)
            compression_ratios = compression_ratios.reshape(len(k_values), len(seeds))
            run_times = np.array(run_times).reshape(compression_ratios.shape)
            run_time_avg = run_times.mean(axis=1)
            mean = compression_ratios.mean(axis=1)
            variance = compression_ratios.var(axis=1)
            seed_col_names = get_column_names(len(seeds), seeds)
            cols = ["K", *seed_col_names, "Avg. Comp. Ratio", "Var. Comp. Ratio", "Avg. Run Time"]
            df = pd.DataFrame([K, *compression_ratios, mean, variance, run_time_avg]).T
            df.columns = cols
            print("Values for Image path:", orig_img_path)
            print(df.to_string())
            print()
        table_cols = ["K", "Seed", "Img Size(KB)", "Comp. Ratio", "Run Time(s)"]
        df = pd.DataFrame(values_table, columns=table_cols)
        print(df.to_string())
    else:
        seed = 131
        img_path = arg_list[1]
        k = int(arg_list[2])
        save_path = arg_list[3]

        k_means_jpg = get_k_means_compressed_img(img_path, k, seed)
        k_means_jpg.save(save_path)
        # Stats
        original_img_info = os.stat(img_path)
        compressed_img_info = os.stat(save_path)

        print("size of image before running K-mean algorithm: ", original_img_info.st_size / 1024, "KB")
        print("size of image after running K-mean algorithm: ", compressed_img_info.st_size / 1024, "KB")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- Total Run Time %s seconds ---" % (round(time.time() - start_time, 3)))
