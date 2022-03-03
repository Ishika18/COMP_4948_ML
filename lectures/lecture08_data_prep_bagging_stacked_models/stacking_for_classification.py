from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy  as np
import pandas as pd

from global_constants import PATH

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Prepare the data.
CSV_DATA = "Social_Network_Ads.csv"

df       = pd.read_csv(PATH + CSV_DATA)
df       = pd.get_dummies(df,columns=['Gender'])
del df['User ID']

X = df.copy()
del X['Purchased']
y = df['Purchased']

def getUnfitModels():
    models = list()
    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=10))
    return models

def evaluateModel(y_test, predictions, model):
    precision = round(precision_score(y_test, predictions),2)
    recall    = round(recall_score(y_test, predictions), 2)
    f1        = round(f1_score(y_test, predictions), 2)
    accuracy  = round(accuracy_score(y_test, predictions), 2)

    print("Precision:" + str(precision) + " Recall:" + str(recall) +\
          " F1:" + str(f1) + " Accuracy:" + str(accuracy) +\
          "   " + model.__class__.__name__)

def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfPredictions[colName] = predictions
    return dfPredictions, models

def fitStackedModel(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Split data into train, test and validation sets.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

# Get base models.
unfitModels = getUnfitModels()

# Fit base and stacked models.
dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
stackedModel          = fitStackedModel(dfPredictions, y_test)

# Evaluate base models with validation data.
print("\n** Evaluate Base Models **")
dfValidationPredictions = pd.DataFrame()
for i in range(0, len(models)):
    predictions = models[i].predict(X_val)
    colName = str(i)
    dfValidationPredictions[colName] = predictions
    evaluateModel(y_val, predictions, models[i])

# Evaluate stacked model with validation data.
stackedPredictions = stackedModel.predict(dfValidationPredictions)
print("\n** Evaluate Stacked Model **")
evaluateModel(y_val, stackedPredictions, stackedModel)
