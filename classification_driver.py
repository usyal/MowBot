# File used to call the classification.py file
# Finds best value for k and collects the metrics for each tested k
# Using this value of k, a classifier model is saved using save_classifier() function in classification.py file
import classification as cl
import os

size_of_grass_images = 0
size_of_non_grass_images = 0
min_validation_error = -1
k = -1
all_metrics_per_k = [] # List containing the metrics for each k-fold {accuracy, f1 score, validation error}

for img in os.listdir("Cleaned-Dataset/Grass"):
    size_of_grass_images += 1

for img in os.listdir("Cleaned-Dataset/Non-Grass"):
    size_of_non_grass_images += 1

max_len = min(size_of_grass_images, size_of_non_grass_images)
i = 2

while (i <= max_len):
    current_metrics = cl.kfold_implementaton(i)
    if (current_metrics[2] < min_validation_error or min_validation_error == -1):
        min_validation_error = current_metrics[2]
        k = i
    all_metrics_per_k.append(current_metrics)
    i += 1


print(all_metrics_per_k)
print(f"Value of k that minimizes the validation error: {k}")