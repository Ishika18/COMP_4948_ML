"""
Best: -0.030820 using {'num_neurons': 60, 'num_layers': 4, 'lr': 0.002, 'initializer': 'he_uniform', 'epochs': 300, 'batch_size': 5}
"""

import json

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import RepeatedKFold

from sklearn.preprocessing import StandardScaler
from global_constants import PATH


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_data():
    return pd.read_csv(f"{PATH}car_prices.csv")


def my_get_dummies(df: pd.DataFrame, cols: list = None, drop_first: bool = False,
                   drop_orig: bool = False) -> pd.DataFrame:
    cols = df.columns.tolist() if cols is None else cols
    for col in cols:
        dummies = pd.get_dummies(df[[col]], columns=[col], drop_first=drop_first)
        df = pd.concat(([df, dummies]), axis=1) if not drop_orig \
            else pd.concat(([df, dummies]), axis=1).drop([col], axis=1)
    return df


def get_model_data():
    df = get_data()
    y = df.copy(True)
    y = y[['current price']]
    x = my_get_dummies(df, cols=['condition', 'rating'], drop_orig=True)
    x = x[['km', 'on road old', 'on road now', 'years',
           'condition_7', 'condition_8', 'condition_9', 'condition_10']]
    return x, y


param_guidelines = {
    'initializer': ['he_normal', 'he_uniform'],
    'batch_size': [5, 10, 15, 25, 50, 100, 150, 200],
    'epochs': [50, 100, 200, 300, 400, 500],
    'num_neurons': [10, 20, 30, 35, 40, 50, 60, 80],
    'num_layers': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'lr': [0.001, 0.0005, 0.002, 0.0015, 0.00075]
}

test_params = {
    'initializer': ['he_uniform'],
    'batch_size': [200],
    'epochs': [50],
    'num_neurons': [10],
    'num_layers': [1],
    'lr': [0.001]
}


def grid_search(x: pd.DataFrame, y: pd.DataFrame, params: dict, random: bool = True,
                filename: str = "random_search_60", iterations: int = 60):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # x_train, y_train = x, y

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled, y_scaled = x_scaler.fit_transform(x_train), y_scaler.fit_transform(y_train)

    n_features = len(x.columns)

    # ---Model Func---#
    def create_model(num_neurons: int = 5, initializer: str = 'he_normal', num_layers: int = 0, lr: int = 0.001):
        model = Sequential()
        model.add(Dense(num_neurons, kernel_initializer=initializer,
                        input_dim=n_features, activation='relu'))
        for i in range(num_layers):
            model.add(Dense(num_neurons, kernel_initializer=initializer,
                            activation='relu'))
        model.add(Dense(1, kernel_initializer=initializer, activation='linear'))

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], optimizer=opt)
        return model

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = KerasRegressor(build_fn=create_model, verbose=0)
    if random:
        grid = RandomizedSearchCV(estimator=model, param_distributions=params, cv=cv,
                                  scoring='neg_root_mean_squared_error', n_iter=iterations)
    else:
        grid = GridSearchCV(estimator=model, param_grid=params, cv=cv,
                            scoring='neg_root_mean_squared_error')

    # ---summarize results---#
    grid_result = grid.fit(x_scaled, y_scaled)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    results = {mean: f'({std:.6f}) {param}' for mean, std, param in zip(means, stds, params)}
    sorted_results = dict(sorted(results.items(), key=lambda x: x[0], reverse=True))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    with open(f'{filename}.json', 'w') as fp:
        fp.write(json.dumps(sorted_results, indent=2))

    with open(f'{filename}.txt', 'a') as fp:
        for k, v in sorted_results.items():
            fp.write(f'{k:.6f}:{v}\n')


if __name__ == '__main__':
    x, y = get_model_data()
    # grid_search(x, y, test_params, filename="Test") # Run this first to just check if it works
    grid_search(x, y, param_guidelines) #actual search
    print("Done")
