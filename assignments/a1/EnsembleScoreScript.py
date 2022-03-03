import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from global_constants import PATH
from tensorflow.python.keras.models import load_model

binary_file_path = './binaries/'

SELECTED_FEATURES = [
    'km', 'on road old', 'on road now',
    'condition_7', 'condition_8', 'condition_9', 'condition_10'
]


def get_csv_data():
    return pd.read_csv(f"{PATH}car_prices.csv")


def get_dummies(df, cols=None, drop_first=False, drop_orig=False):
    cols = df.columns.tolist() if cols is None else cols
    for col in cols:
        dummies = pd.get_dummies(df[[col]], columns=[col], drop_first=drop_first)
        df = pd.concat(([df, dummies]), axis=1) if not drop_orig \
            else pd.concat(([df, dummies]), axis=1).drop([col], axis=1)
    return df


def prepare_model_data():
    df = get_csv_data()
    x = get_dummies(df, cols=['condition'], drop_orig=True)
    return x[SELECTED_FEATURES]


def evaluate_model(y_test, y_pred, mode_name):
    print(f"\n\n#### Model {mode_name} ####\n")
    print(f'RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def predict_with_base_models(models, y_scaler, X_test):
    predictions = pd.DataFrame()

    for title, (model, X_scaler) in models.items():
        x_test_scaled = X_scaler.transform(X_test[SELECTED_FEATURES])
        y_pred_scaled = model.predict(x_test_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        predictions[title] = np.stack(y_pred, axis=1)[0].tolist()

    return pd.DataFrame(predictions)


def predict(df):
    # load stacked model
    path = f'{binary_file_path}stacked_model.pkl'
    with open(path, 'rb') as file:
        stacked_model = pickle.load(file)

    # load y scaler
    path = f'{binary_file_path}y_scaler.pkl'
    with open(path, 'rb') as file:
        y_scaler = pickle.load(file)

    models = {}
    for title in ["model0", "model1", "model2"]:
        res = []

        # load model saved
        path = f'{binary_file_path}{title}.h5'
        res.append(load_model(path))

        # load x scaler
        path = f'{binary_file_path}x_scaler.pkl'
        with open(path, 'rb') as file:
            res.append(pickle.load(file))

        models[title] = (res[0], res[1])

    x = df[SELECTED_FEATURES]
    x_test = predict_with_base_models(models, y_scaler, x)
    y_pred = stacked_model.predict(x_test)
    return pd.DataFrame({'current price': np.stack(y_pred, axis=1)[0].tolist()})


def predict_car_prices(df) -> None:
    car_prices_predictions = predict(df)

    if car_prices_predictions is not None:
        car_prices_predictions.to_csv(f"car_prices_predictions.csv", index=False)


def main() -> None:
    df = prepare_model_data()
    predict_car_prices(df)


if __name__ == '__main__':
    main()
