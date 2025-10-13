# File used to call the classification.py file
# Finds best value for k and collects the metrics for each tested k
# Using this value of k, a classifier model is saved using save_classifier() function in classification.py file
import classification
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

i = 2

for i in range(5, 24): # Range of 5-23 typically gives best k
    current_metrics = classification.kfold_implementaton(i)
    if (current_metrics[2] < min_validation_error or min_validation_error == -1):
        min_validation_error = current_metrics[2]
        k = i
    all_metrics_per_k.append(current_metrics)
    i += 1

for i in range(5, 24): 
    print(f"k = {i}. {all_metrics_per_k[i - 5]}" )
print(f"Value of k that minimizes the validation error: {k}")

# Saves the trained classifier
classification.save_classifier(k)