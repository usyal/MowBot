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
    pass


features = []
labels = []

for label, folder in enumerate(["Grass", "Not-Grass"]):
    path = os.path.join("Dataset", folder)
    for img_file in os.listdir(path):
        img_location = os.path.join(path, img_file)
        features.append(feature_extractor(img_location))
        labels.append(label)

features = np.array(features)
labels = np.array(labels)