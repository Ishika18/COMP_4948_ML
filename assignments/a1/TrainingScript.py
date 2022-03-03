import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from global_constants import PATH

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pickle as pkl

SELECTED_FEATURES = [
    'km', 'on road old', 'on road now',
    'condition_7', 'condition_8', 'condition_9', 'condition_10'
]
# NOTE
# PLEASE CREATE THE FOLDER IF NOT ALREADY CREATED
# YOU CAN CHANGE THE CONSTANT DIRECTLY
binary_file_path = './binaries/'


def get_data():
    return pd.read_csv(f"{PATH}car_prices.csv")


def get_dummies(df, cols=None, drop_first=False, drop_orig=False):
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
    x = get_dummies(df, cols=['condition'], drop_orig=True)
    x = x[SELECTED_FEATURES]
    return x, y


def save_scalers(x_scaler=None, y_scaler=None):
    if x_scaler is None and y_scaler is None:
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
    pkl.dump(x_scaler, open(f'{binary_file_path}x_scaler.pkl', 'wb'))
    pkl.dump(y_scaler, open(f'{binary_file_path}y_scaler.pkl', 'wb'))


def create_model(**kwargs):
    model = Sequential()
    model.add(Dense(kwargs['num_neurons'], kernel_initializer=kwargs['initializer'],
                    input_dim=kwargs['input_dim'], activation='relu'))
    for i in range(kwargs['num_layers']):
        model.add(Dense(kwargs['num_neurons'] * 2 // 3 + 1, kernel_initializer=kwargs['initializer'],
                        activation='relu'))
    model.add(Dense(1, kernel_initializer=kwargs['initializer'], activation='linear'))

    opt = tf.keras.optimizers.Adam(learning_rate=kwargs['lr'])
    model.compile(loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], optimizer=opt)
    return model


def plot_loss_and_rmse(history, model_name):
    # Plot loss learning curves.
    plt.title(f'{model_name} Plot', pad=-40)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['root_mean_squared_error'], label='rmse')
    plt.legend()
    plt.show()


def load_scalers():
    x_scaler = pkl.load(open(f'{binary_file_path}x_scaler.pkl', 'rb'))
    y_scaler = pkl.load(open(f'{binary_file_path}y_scaler.pkl', 'rb'))
    return x_scaler, y_scaler


def evaluate_model(model, x_test, y_test, model_name):
    print(f"#### {model_name} ####")
    predictions = model.predict(x_test)
    x_scaler, y_scaler = load_scalers()
    unscaled_pred = y_scaler.inverse_transform(predictions)
    rmse = np.sqrt(mean_squared_error(y_test, unscaled_pred))
    print('Root Mean Squared Error:', rmse)


def early_stopping(x_train, x_test, y_train, y_test,
                   model_params, filename="model", model_name=""):
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100)
    mc = ModelCheckpoint(f'{filename}.h5', monitor='loss', mode='min', verbose=1,
                         save_best_only=True)

    model = create_model(**model_params)
    history = model.fit(x_train, y_train, epochs=4000, verbose=1,
                        callbacks=[es, mc])

    plot_loss_and_rmse(history, model_name)
    evaluate_model(model, x_test, y_test, model_name)
    return model


def build_models(X_train, X_test, y_train, y_test):
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled, y_scaled = x_scaler.fit_transform(X_train), y_scaler.fit_transform(y_train.copy(True))
    x_test_scaled = x_scaler.fit_transform(X_test)
    save_scalers(x_scaler, y_scaler)

    model_params_list = [
        {'num_neurons': 60, 'num_layers': 4, 'lr': 0.002,
         'initializer': 'he_uniform', 'epochs': 300, 'batch_size': 5, 'input_dim': len(X_train.columns.tolist())},
        {'num_neurons': 90, 'num_layers': 8, 'lr': 0.00075,
         'initializer': 'he_uniform', 'epochs': 550, 'batch_size': 10, 'input_dim': len(X_train.columns.tolist())},
        {'num_neurons': 10, 'num_layers': 2, 'lr': 0.01,
         'initializer': 'he_uniform', 'epochs': 600, 'batch_size': 10, 'input_dim': len(X_train.columns.tolist())},
    ]

    models = []
    for i, model_params in enumerate(model_params_list):
        models.append(early_stopping(x_scaled, x_test_scaled, y_scaled, y_test, model_params,
                                     f"{binary_file_path}model{i}", f"Model {i}"))

    return models


def build_stacked_model(models, x_test, x_val, y_test, y_val):
    df_predictions = pd.DataFrame()
    df_validation_predictions = pd.DataFrame()
    for i, model in enumerate(models):
        x_scaler, y_scaler = load_scalers()
        x_test_scaled = x_scaler.transform(x_test.copy(True)[SELECTED_FEATURES])
        x_val_scaled = x_scaler.transform(x_val.copy(True)[SELECTED_FEATURES])
        predictions = model.predict(x_test_scaled)
        validation_predictions = model.predict(x_val_scaled)
        df_predictions[str(i)] = np.stack(predictions, axis=1)[0].tolist()
        df_validation_predictions[str(i)] = np.stack(validation_predictions, axis=1)[0].tolist()

    stacked_model = Ridge()
    stacked_model.fit(df_predictions, y_test)

    # Save model into a binary file
    pkl.dump(stacked_model, open('%sstacked_model.pkl' % binary_file_path, 'wb'))

    print(f"#### Stacked Model ####")
    stacked_predictions = stacked_model.predict(df_validation_predictions)

    rmse = np.sqrt(mean_squared_error(y_val, stacked_predictions))
    print('Root Mean Squared Error:', rmse)


def main():
    X = get_model_data()[0]
    y = get_model_data()[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    models = build_models(X_train, X_test, y_train, y_test)
    build_stacked_model(models, X_test, X_train, y_test, y_train)


if __name__ == '__main__':
    main()
