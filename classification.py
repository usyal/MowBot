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
