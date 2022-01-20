import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from global_constants import PATH
from tensorflow.keras.optimizers import Adam

FILE = "heart_disease.csv"
data = pd.read_csv(PATH + FILE)
x_data = data.drop("target", axis=1)
y_values = data["target"]

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_values, test_size=0.3, random_state=42
)

# Stochastic gradient descent models are sensitive to differences
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_trainScaled = scaler.transform(X_train)
X_testScaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_trainScaled, y_train)
lr_pred = clf.predict(X_testScaled)

print("Accuracy:{} ".format(clf.score(X_testScaled, y_test) * 100))
print("Error Rate:{} ".format((1 - clf.score(X_testScaled, y_test)) * 100))

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(y_test, lr_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ', metrics.accuracy_score(y_test, lr_pred))
print("\nConfusion Matrix")
print(confusion_matrix)

COLUMN_DIMENSION = 1
#######################################################################
# Part 2
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# shape() obtains rows (dim=0) and columns (dim=1)
n_features = X_trainScaled.shape[COLUMN_DIMENSION]


# Define the model.

def create_model(neurons=1):
    model = Sequential()
    model.add(Dense(neurons, kernel_initializer='uniform',
                    input_dim=n_features, activation='tanh'))
    model.add(Dense(1, kernel_initializer='uniform'))
    # Use Adam optimizer with the given learning rate
    opt = Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


### Grid Building Section #######################
model   = KerasClassifier(build_fn=create_model, epochs=90, batch_size=60, verbose=1)
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid       = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#################################################

grid_result = grid.fit(X_train, y_train)



# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
