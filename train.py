import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import linear_model, svm, tree, ensemble
from xgboost import XGBRegressor


def read_data():
    data = pd.read_csv('data/train.csv')  # read pre-processed dataset
    y = data.y  # extract labels
    data.drop('y', axis=1, inplace=True)

    # Create 10-fold cross-validation.
    data_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)

    # Machine Learning Models for initial comparisons.
    ML_models = {
        # Linear models:
        'linear_regressor': linear_model.LinearRegression(),
        'ridge_regressor': linear_model.Ridge(alpha=0.1),
        'LASSO': linear_model.Lasso(alpha=0.1),
        'elastic_net': linear_model.ElasticNet(alpha=0.1),

        # Decision Tree Regression model:
        'tree_regressor': tree.DecisionTreeRegressor(),

        # SVM model:
        'svr': svm.LinearSVR(),

        # ensemble methods:
        'boosting_regressor': ensemble.GradientBoostingRegressor(criterion='mse'),
        'random_forest_regressor': ensemble.RandomForestRegressor(),

        # xgboost model:
        'xgb_regressor': XGBRegressor()
    }

    return data, y, data_split, ML_models


def train():
    x, y, data_split, ML_models = read_data()
    y = np.array(y)

    labels = ['Machine Learning Model', 'Parameters', 'Train Score Mean', 'Test Score Mean', 'Training Time']
    result = pd.DataFrame(columns=labels)
    predictions = pd.DataFrame({'y': y})

    i = 0
    for model in ML_models:
        # For each model, train and test on the dataset and save the comparison results.
        result.loc[i, 'Machine Learning Model'] = model
        result.loc[i, 'Parameters'] = str(ML_models[model].get_params())

        # Model training with cross-validation.
        cv_results = model_selection.cross_validate(ML_models[model], x, y, cv=data_split,
                                                    scoring='neg_mean_squared_error')
        result.loc[i, 'Train Score Mean'] = np.sqrt(-cv_results['train_score'].mean())
        result.loc[i, 'Test Score Mean'] = np.sqrt(-cv_results['test_score'].mean())
        result.loc[i, 'Training Time'] = cv_results['fit_time'].mean()

        ML_models[model].fit(x, y)
        predictions[model] = ML_models[model].predict(x)  # save predictions
        i += 1

    result.sort_values(by=['Test Score Mean'], ascending=True, inplace=True)
    result.to_csv('results/result.csv', index=False)
    predictions.to_csv('results/predictions.csv', index=False)
    print(result)


if __name__ == "__main__":
    train()
