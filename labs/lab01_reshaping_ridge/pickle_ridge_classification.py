import pandas  as pd
import numpy   as np
from   sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from   sklearn                 import metrics
import statsmodels.api        as sm
from global_constants import PATH
import pickle

FILE  = 'heart_disease.csv'
df = pd.read_csv(PATH + FILE)
print(df)

# Separate into x and y values.
X = df[['age',
        'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']

# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Show chi-square scores for each feature.
# There is 1-degree freedom since 1 predictor during feature evaluation.
# Generally, >=3.8 is good)
test      = SelectKBest(score_func=chi2, k=3)
chiScores = test.fit(X, y) # Summarize scores
np.set_printoptions(precision=3)
print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

# Re-assign X with significant columns only after chi-square test.
X = df[['age',
        'sex', 'cp', 'trestbps', 'chol', 'fbs',  'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal']]

# Split data.
X_train,X_test,y_train,y_test = train_test_split(
        X, y, test_size=0.25,random_state=0)

# Stochastic gradient descent models are sensitive to differences
# in scale so a StandardScaler is usually used.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_trainScaled = scaler.transform(X_train)
X_testScaled  = scaler.transform(X_test)
X_trainScaled = sm.add_constant(X_trainScaled)
X_testScaled  = sm.add_constant(X_testScaled)

print("\nRidge Classifier")

clf = RidgeClassifier(solver='auto')
clf.fit(X_trainScaled, y_train)

y_pred = clf.predict(X_testScaled)
print(y_pred)

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(y_test, y_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix)

# save model to file
pickle.dump(clf, open("myRidgeModel.dat", "wb"))

# load model from file
loaded_model = pickle.load(open("myRidgeModel.dat", "rb"))
ridgePredictions = loaded_model.predict(X_testScaled)
print(ridgePredictions)
