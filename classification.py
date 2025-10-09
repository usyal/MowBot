# AI model for classification using sklearn library
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

    gray = (color.rgb2gray(img) * 255).astype(np.uint8) # Corrects values for images to be exactly 8 bits by scaling so no floats are used
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

for label, img in enumerate(["Grass", "Non-Grass"]):
    path = os.path.join("Cleaned-Dataset", img)
    for img_file in os.listdir(path):
        img_location = os.path.join(path, img_file)
        features.append(feature_extractor(img_location))
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

for i in range(len(features)):
    print(features[i], labels[i])
    print("................................")


# Metrics used for evaluation of K-Fold
def accuracy_metric(validation_results, labels):
    TP = np.sum((validation_results == 1) & (labels == 1))
    TN = np.sum((validation_results == 0) & (labels == 0))
    FP = np.sum((validation_results == 1) & (labels == 0))
    FN = np.sum((validation_results == 0) & (labels == 1))

    if TP == 0 and TN == 0 and FN == 0 and FP == 0:
        return 0.0

    return float((TP + TN) / (TP + TN + FP + FN))

def f1_score(validation_results, labels):
    TP = np.sum((validation_results == 1) & (labels == 1))
    FP = np.sum((validation_results == 1) & (labels == 0))
    FN = np.sum((validation_results == 0) & (labels == 1))

    if not (TP + FP > 0) or not (TP + FN > 0):
        return 0.0
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return float((2 * precision * recall) / (precision + recall))    

def get_incorrect_predictions(validation_results, labels):
    num = 0
    for i in range(len(validation_results)):
        if labels[i] != validation_results[i]:
            num += 1
    return num / len(validation_results)

# K-Fold implementation
skf = kfold(n_splits = 4, shuffle = True, random_state = 10) # n_splits is value for k, shuffle and random_state work to randomize the samples in each fold
k_fold_metrics = []
k_fold_validation_errors = []

for train_img, test_img in skf.split(features, labels): # skf.split() generates train and test indexes for each fold
    training_features = features[train_img]
    testing_features = features[test_img]

    training_labels = labels[train_img]
    testing_labels = labels[test_img]

    classifier = rfc(random_state = 10) # Random Forest Classifier - generates a new classifer model which is untrained and random_state is a seed to get consistent results
    # fit() function that builds decision trees to determine what prediction to make for classification. 
    # The trained model/trees are stored in the classifier object
    classifier.fit(training_features, training_labels) 

    # Validating whether the model is able to predict correctly or not on the testing data
    validation_results = classifier.predict(testing_features)

    # Current evaluation array for current fold will store the metrics for accuracy, f1 score, and validation error
    current_evaluation = []
    current_evaluation.append(accuracy_metric(validation_results, testing_labels)) # Accuracy Score
    current_evaluation.append(f1_score(validation_results, testing_labels)) # F1 score
    k_fold_metrics.append(current_evaluation)

    k_fold_validation_errors.append(get_incorrect_predictions(validation_results, testing_labels)) # Error for each fold is saved in this list


for metric in k_fold_metrics:
    print(metric)
    print(".....................")

# Validation Error for the K-fold
print(np.sum(k_fold_validation_errors) / skf.get_n_splits())

# Once model is trained, model can be saved using joblib or pickle library, but joblib is more efficient for models using larger arrays 