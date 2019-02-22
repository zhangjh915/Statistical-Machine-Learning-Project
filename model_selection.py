"""
This file is used to tune the parameters of the models.
From the result generated from 'train.py', the models with the two lowest RMSE values are selected:
                Linear Regressor        &       Gradient Boosting Regressor.
Since there is no parameter to be tuned for linear regressor, only Gradient Boosting Regressor is included here.
"""
import pickle
import numpy as np
from sklearn import model_selection
from sklearn import linear_model, ensemble
from sklearn.model_selection import GridSearchCV
from train import read_data


def parameter_tuning():
    # Conduct grid search to tune the parameters for GBR for the best performance.
    x, y, _, _ = read_data()
    y = np.array(y)

    param_candidates = {
        'learning_rate': [0.1, 1],
        'n_estimators': [100, 200, 500],
        'min_samples_leaf': [1, 2, 5],
        'max_depth': [3, 5, 10],
        },
    regressor = ensemble.GradientBoostingRegressor(criterion='mse')
    gsearch = GridSearchCV(regressor, param_grid=param_candidates, scoring='neg_mean_squared_error', cv=5)
    gsearch.fit(x, y)
    print(gsearch.best_params_)  # {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 5, 'n_estimators': 200}


def save_model():
    # Train, validate and save the two models for Linear Regressor and GBR.
    x, y, _, _ = read_data()
    y = np.array(y)

    # Evaluate performance.
    linear_regressor = linear_model.LinearRegression()
    gradient_boost_regressor = ensemble.GradientBoostingRegressor(
        learning_rate=0.1, n_estimators=200, min_samples_leaf=5, max_depth=5, criterion='mse')
    # gradient_boost_regressor = ensemble.GradientBoostingRegressor(criterion='mse')
    data_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
    performance_linear_cv = model_selection.cross_validate(linear_regressor, x, y, cv=data_split,
                                                           scoring='neg_mean_squared_error')
    performance_boosting_cv = model_selection.cross_validate(gradient_boost_regressor, x, y, cv=data_split,
                                                             scoring='neg_mean_squared_error')

    performance_linear, performance_boosting = {}, {}
    performance_linear['Train Score Mean'] = np.sqrt(-performance_linear_cv['train_score'].mean())
    performance_linear['Test Score Mean'] = np.sqrt(-performance_linear_cv['test_score'].mean())
    performance_linear['Training Time'] = performance_linear_cv['fit_time'].mean()
    performance_boosting['Train Score Mean'] = np.sqrt(-performance_boosting_cv['train_score'].mean())
    performance_boosting['Test Score Mean'] = np.sqrt(-performance_boosting_cv['test_score'].mean())
    performance_boosting['Training Time'] = performance_boosting_cv['fit_time'].mean()
    print(performance_linear)
    print(performance_boosting)

    # Fit dataset.
    linear_regressor.fit(x, y)
    gradient_boost_regressor.fit(x, y)

    # Save models.
    with open('save/linear_regressor.pickle', 'wb') as f:
        pickle.dump(linear_regressor, f)
    with open('save/gradient_boost_regressor.pickle', 'wb') as f:
        pickle.dump(gradient_boost_regressor, f)


if __name__ == "__main__":
    save_model()
