from pydataset import data
import pandas  as pd
from sklearn.ensemble        import BaggingClassifier, \
         ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC

# Get the housing data
df = data('Housing')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\n*** Before data prep.")
print(df.head(5))

# Convert continues price variable into evenly distributed categories.
df['price'] = pd.qcut(df['price'], 3, labels=[0,1,2]).cat.codes

# Show new price variable cateogories.
print("\nNewly categorized target (price) values")
print(df['price'].value_counts())

def convertToBinaryValues(df, columns):
    for i in range(0, len(columns)):
        df[columns[i]] = df[columns[i]].map({'yes': 1, 'no': 0})
    return df

df = convertToBinaryValues(df, ['driveway', 'recroom', \
            'fullbase', 'gashw', 'airco', 'prefarea'])

# Split into two sets
y = df['price']
X = df.drop('price', 1)

# Show prepared data.
print("\n*** X")
print(X.head(5))

print("\n*** y")
print(y.head(5))


# Create classifiers
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()
lr = LogisticRegression(fit_intercept=True, solver='liblinear')

# Build array of classifiers.
classifierArray   = [rf, et, knn, svc, rg, lr]

def showStats(classifier, scores):
    print(classifier + ":    ", end="")
    strMean = str(round(scores.mean(),2))

    strStd  = str(round(scores.std(),2))
    print("Mean: "  + strMean + "   ", end="")
    print("Std: " + strStd)

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

def evaluateModel(model, X_test, y_test, title):
    print("\n*** " + title + " ***")
    predictions = model.predict(X_test)
    accuracy    = metrics.accuracy_score(y_test, predictions)
    recall      = metrics.recall_score(y_test, predictions, average='weighted')
    precision   = metrics.precision_score(y_test, predictions, average='weighted')
    f1          = metrics.f1_score(y_test, predictions, average='weighted')

    print("Accuracy:  " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall:    " + str(recall))
    print("F1:        " + str(f1))

# Search for the best classifier.
for clf in classifierArray:
    modelType = clf.__class__.__name__

    # Create and evaluate stand-alone model.
    clfModel    = clf.fit(X_train, y_train)
    evaluateModel(clfModel, X_test, y_test, modelType)

    # max_features means the maximum number of features to draw from X.
    # max_samples sets the percentage of available data used for fitting.
    bagging_clf = BaggingClassifier(clf, max_samples=0.4, max_features=3,
                                    n_estimators=10)
    baggedModel = bagging_clf.fit(X_train, y_train)
    evaluateModel(baggedModel, X_test, y_test, "Bagged: " + modelType)
