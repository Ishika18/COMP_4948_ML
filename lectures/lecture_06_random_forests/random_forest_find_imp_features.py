# Import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

NUM_IMP_COLS = 3

# Load dataset
iris = datasets.load_iris()

# Creating a DataFrame of given iris dataset.
import pandas as pd
data = pd.DataFrame({
    'sepal length': iris.data[:, 0],
    'sepal width': iris.data[:, 1],
    'petal length': iris.data[:, 2],
    'petal width': iris.data[:, 3],
    'species': iris.target
})
iris['target_names']
feature_list = ['sepal length', 'sepal width', 'petal length', 'petal width']
print(data.head())

# Import train_test_split function

X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y = data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Import Random Forest Model


# Create a Gaussian Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the model using the training sets y_pred=rf.predict(X_test)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation

# Model Accuracy, how often is the classifier correct?
print("Normal Model Accuracy")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# ---Viewing Feature Importance--- #
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
print("\nFeature Importance")
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# --- Forest with only top features --- #
import numpy as np
# New random forest with only the three most important variables
rf_most_important = RandomForestClassifier(n_estimators=100, random_state=0)

# Extract the three most important features
important_cols = [feature_importances[x][0] for x in range(0, NUM_IMP_COLS)]
train_important = X_train[important_cols]
test_important = X_test[important_cols]

# Train the random forest
rf_most_important.fit(train_important, y_train)

# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - y_test)

# Display the performance metrics
print(f"\nAfter Selecting top {NUM_IMP_COLS}")
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_test))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')