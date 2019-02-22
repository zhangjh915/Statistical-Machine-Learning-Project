import pickle
import pandas as pd

from preprocess import read_data, process_data, fill_data


data = read_data('test')
data = process_data(data, filled=False, data_type='test')
data = fill_data(data)
data = process_data(data, filled=True, data_type='test')
data.to_csv('data/test.csv', index=False)

with open('save/linear_regressor.pickle', 'rb') as f:
    model = pickle.load(f)
    linear_regressor_predictions = model.predict(data)
    linear_regressor_predictions = pd.DataFrame(linear_regressor_predictions.reshape(1, 10000))
    linear_regressor_predictions.to_csv("results/linear_regressor_predictions.csv", header=False, index=False)
with open('save/gradient_boost_regressor.pickle', 'rb') as f:
    model = pickle.load(f)
    gradient_boost_regressor_predictions = model.predict(data)
    gradient_boost_regressor_predictions = pd.DataFrame(gradient_boost_regressor_predictions.reshape(1, 10000))
    gradient_boost_regressor_predictions.to_csv("results/gradient_boost_regressor_predictions.csv", header=False, index=False)
