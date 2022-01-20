import pandas as pd
from sklearn.model_selection import train_test_split
from global_constants import PATH
from tensorflow.keras.optimizers import Adam

CSV_DATA = "housing.data"
df = pd.read_csv(PATH + CSV_DATA, header=None)

# Show all columns.
pd.set_option('display.max_columns', None)

# Increase number of columns that display on one line.
pd.set_option('display.width', 1000)

print(df.head())
print(df.tail())
print(df.describe())

dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
y = dataset[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.2, random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

n_features = X_train.shape[1]


# define base model
def create_model(init_mode='uniform'):
    model = Sequential()
    model.add(Dense(13, kernel_initializer=init_mode,
                    input_dim=n_features, activation='relu'))
    model.add(Dense(1, kernel_initializer=init_mode))

    opt = Adam(lr=0.005)
    # Compile model
    model.compile(loss='mse', metrics=['accuracy'], optimizer=opt)
    return model


### Grid Building Section #######################
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=1)

# Define the grid search parameters.
init_mode  = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
              'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
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
