# AI model for classification
import sklearn as sl
# Stratified K fold makes sure the proportions are equal for testing/training data
from sklearn.model_selection import StratifiedKFold as kfold
# skimage allows for feature extraction
from skimage import io, color, feature
# RandomForestClassifier perform classification by construction decision trees
# Commonly implemented with K Fold 
from sklearn.ensemble import RandomForestClassifier as rfc
import numpy as np
import os

def feature_extractor(img_location):
    img = io.imread(img_location)
    hsv = color.rgb2hsv(img) # Hue, Saturation, vibrance 
    green = (hsv[:, :, 0] >= 0.2) & (hsv[:, :, 0] <= 0.55) # Typical range in which real world grass colour falls under
    green_amount = np.sum(green) / (img.shape[0] * img.shape[1]) # rows * cols (Normalizing the green colour ratio)

    gray = color.rgb2gray(img)
    # P is the number of neighbours to look at and R is the radius each neighbour is away at
    # For each pixel, local_binary_pattern computes a P bit binary number (neightbour >= current pixel intensity then 1, else 0)
    # This is useful for analyzing the textures based of intensity
    bp = feature.local_binary_pattern(gray, P = 10, R = 1)

    # Histogram tracks how many pixels have each binary pattern 
    # Bins is used to split the patterns into buckets and the range is from 0 to bins
    # This simplifies the patterns by excluding many extra patterns
    bp_histogram, _ = np.histogram(bp, bins = 10, range = (0, 10))

    # Scaling the values so we don't need to worry about large number of binary patterns
    bp_histogram = bp_histogram / np.sum(bp_histogram)

    # Returning an array of green ratio for each image and textures
    return np.concatenate([[green_amount], bp_histogram])

features = []
labels = []

for label in (["Grass", "Non-Grass"]):
    path = os.path.join("Cleaned-Dataset", label)
    for img_file in os.listdir(path):
        img_location = os.path.join(path, img_file)
        features.append(feature_extractor(img_location))
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

print(features, labels)