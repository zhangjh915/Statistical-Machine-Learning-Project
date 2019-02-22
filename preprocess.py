import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def read_data(data_type='train'):
    # Load training or test dataset.
    if data_type == 'train':
        data = pd.read_csv('data/cryoocyte_3_regression_train.csv')
    elif data_type == 'test':
        data = pd.read_csv('data/cryoocyte_3_regression_test.csv')
    else:
        raise ValueError('Unknown Data Type: %s' % data_type)
    print('Dimension for %s set is: %s' % (data_type, data.shape))
    # Dimension for train set is: (40000, 116) with y
    # Dimension for test set is: (10000, 115) without y
    return data


def check_data(data):
    # Check data non-numerical features and nan rate.

    # Check numerical and object variables.
    num_var = data.applymap(np.isreal).all(0)
    obj_var = {}
    for index, value in num_var.items():
        if not value:
            obj_var[index] = data.loc[0, index]
    # obj_var = {'x44': '0.0%', 'x50': 'tuesday', 'x59': '$-1832.38', 'x63': 'Orang', 'x65': 'D', 'x95': 'Aug'}

    # Calculate nan rate for each feature and y and create a data frame to store the information.
    nan_rate = {}
    for x in data.columns:
        nan_rate[x] = 100 * data[x].isnull().sum() / len(data[x])
    nan_rate_df = pd.DataFrame(list(nan_rate.values()), index=nan_rate.keys(), columns=['nan_rate'])
    nan_rate_df['data_type'] = data.dtypes
    # max(nan_rate_df['nan_rate']) = 0.0725%; nan_rate = 0 for y

    return nan_rate_df


def process_data(data, filled=False, data_type='train'):
    # Change the non-numerical quantitative features to numeric values.

    if data_type == 'train':
        num_sample = 40000
    elif data_type == 'test':
        num_sample = 10000
    else:
        raise ValueError('Unknown Data Type: %s' % data_type)

    if not filled:  # before filling the nan values
        # Change x44(percentage with %) and x59(price with $) to numerical values.
        for i in range(num_sample):
            x44 = data.loc[i, 'x44']
            x59 = data.loc[i, 'x59']
            try:
                data.loc[i, 'x44'] = float(x44[:-1]) * 0.01
            except TypeError:  # nan values
                pass
            try:
                data.loc[i, 'x59'] = float(x59[1:])
            except TypeError:  # nan values
                pass
        data = data.astype({"x44": float, "x59": float})

        # Replace the nan values of week and month features with their modes respectively.
        for x in ['x50', 'x95']:
            data[x].fillna(data[x].mode()[0], inplace=True)

        # Use sine transformation on x50(week) and x95(month) variables to keep their temporal relationship.
        x50_sin = pd.Series()
        x50_cos = pd.Series()
        x95_sin = pd.Series()
        x95_cos = pd.Series()
        for i in range(num_sample):
            x50 = data.loc[i, 'x50']
            x95 = data.loc[i, 'x95']
            week_dict = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 5, 'friday': 4, 'sat': 5, 'sun': 6}
            month_dict = {'January': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
                          'July': 6, 'Aug': 7, 'sept.': 8, 'Oct': 9, 'Nov': 10, 'Dev': 11}
            try:
                x50_sin.at[i] = np.sin(week_dict[x50] * (2 * np.pi / 7))
                x50_cos.at[i] = np.cos(week_dict[x50] * (2 * np.pi / 7))
            except KeyError:  # nan values
                x50_sin.at[i] = np.nan
                x50_cos.at[i] = np.nan
            try:
                x95_sin.at[i] = np.sin(month_dict[x95] * (2 * np.pi / 12))
                x95_cos.at[i] = np.cos(month_dict[x95] * (2 * np.pi / 12))
            except KeyError:  # nan values
                x95_sin.at[i] = np.nan
                x95_cos.at[i] = np.nan
        data['x50_sin'] = x50_sin
        data['x50_cos'] = x50_cos
        data['x95_sin'] = x95_sin
        data['x95_cos'] = x95_cos
        data.drop('x50', axis=1, inplace=True)
        data.drop('x95', axis=1, inplace=True)

    else:  # after filling the nan values
        # One-hot encoding of categorical features.
        one_hot_variables = pd.get_dummies(data[['x63', 'x65']])

        data.drop(['x63', 'x65'], axis=1, inplace=True)
        data = pd.concat([one_hot_variables, data], axis=1)

    return data


def fill_data(data):
    # Fill nan values of the dataset.

    # Replace the nan values of the categorical features with their modes respectively.
    for x in ['x63', 'x65']:
        data[x].fillna(data[x].mode()[0], inplace=True)

    # Replace the nan values of the numerical features with the means.
    nan_rate_df = check_data(data)
    numerical_features = list(nan_rate_df.loc[nan_rate_df.data_type == 'float', ].index)
    for x in numerical_features:
        data[x].fillna(data[x].mean(), inplace=True)

    return data


def count_cat(data):
    # This function is not called but was used to obtain the lists of categorical variables.
    cat50 = {}
    cat63 = {}
    cat65 = {}
    cat95 = {}
    for i in range(40000):
        x50 = data.loc[i, 'x50']
        x63 = data.loc[i, 'x63']
        x65 = data.loc[i, 'x65']
        x95 = data.loc[i, 'x95']
        if x50 not in cat50:
            cat50[x50] = 0
        else:
            cat50[x50] += 1
        if x63 not in cat63:
            cat63[x63] = 0
        else:
            cat63[x63] += 1
        if x65 not in cat65:
            cat65[x65] = 0
        else:
            cat65[x65] += 1
        if x95 not in cat95:
            cat95[x95] = 0
        else:
            cat95[x95] += 1
    # cat50 = {'tuesday': 18114, 'monday': 6534, 'wednesday': 12552, 'thursday': 2162, 'sun': 518,
    #          'friday': 82, nan: 15, 'sat': 15}
    # cat63 = {'Orang': 14538, 'Yellow': 23454, 'red': 351, 'blue': 1638, nan: 14}
    # cat65 = {'D': 6224, 'B': 32715, 'A': 1029, nan: 28}
    # cat95 = {'Aug': 5672, 'May': 7371, 'July': 9877, 'Apr': 2934, 'Jun': 10933, 'sept.': 1864,
    #          'Mar': 751, 'Feb': 102, 'Oct': 408, 'Nov': 40, nan: 21, 'January': 11, 'Dev': 3}


def main(plot=False):
    data = read_data('train')
    nan_rate_df = check_data(data)
    data = process_data(data, filled=False)
    data = fill_data(data)
    data = process_data(data, filled=True)
    data.to_csv('data/train.csv', index=False)

    # Plot scatter plots for each feature.
    if plot:
        data = pd.read_csv('train.csv')
        nan_rate_df = check_data(data)

        numerical_features = list(nan_rate_df.loc[nan_rate_df.data_type == 'float', ].index)
        for x in ['y', 'x50_sin', 'x50_cos', 'x95_sin', 'x95_cos']:
            numerical_features.remove(x)

        fig, axs = plt.subplots(11, 11, figsize=(20, 20))
        for i in range(len(numerical_features)):
            sns.scatterplot(x=numerical_features[i], y='y', data=data[[numerical_features[i], 'y']], ax=axs[i//11][i%11])
        plt.subplots_adjust(hspace=0.5)
        plt.savefig('results/scatter_plots.png')


if __name__ == "__main__":
    main(plot=False)
