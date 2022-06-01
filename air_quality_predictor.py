import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from math import sqrt


def wd_to_numeric(x):
    """
    Transform wind direction to numerical value
    """
    if x == 'N':
        return 0
    elif x == 'NNE':
        return 1
    elif x == 'NE':
        return 2
    elif x == 'ENE':
        return 3
    elif x == 'E':
        return 4
    elif x == 'ESE':
        return 5
    elif x == 'SE':
        return 6
    elif x == 'SSE':
        return 7
    elif x == 'S':
        return 8
    elif x == 'SSW':
        return 9
    elif x == 'SW':
        return 10
    elif x == 'WSW':
        return 11
    elif x == 'W':
        return 12
    elif x == 'WNW':
        return 13
    elif x == 'NW':
        return 14
    elif x == 'NNW':
        return 15
    else:
        print("Error")


# Import the datasets
# X1, Y1 are the meteorological data and the target variable used to train our neural net
X1 = pd.read_csv("Datasets/X1.csv")
Y1 = pd.read_csv("Datasets/Y1.csv", header=None, names=['PM2.5'])

# X2 is the meteorological data for which the target variable is not known
X2 = pd.read_csv("Datasets/X2.csv")

# Data pre-processing
# Dates and wind direction are cyclical feature. To account for this, we use sine and cosine transformations.
X1['hour_sin'] = np.sin(2 * np.pi * X1['hour'] / 23.0)
X1['hour_cos'] = np.cos(2 * np.pi * X1['hour'] / 23.0)

X2['hour_sin'] = np.sin(2 * np.pi * X2['hour'] / 23.0)
X2['hour_cos'] = np.cos(2 * np.pi * X2['hour'] / 23.0)

X1['day_sin'] = np.sin(2 * np.pi * X1['day'] / 30.5)
X1['day_cos'] = np.cos(2 * np.pi * X1['day'] / 30.5)

X2['day_sin'] = np.sin(2 * np.pi * X2['day'] / 30.5)
X2['day_cos'] = np.cos(2 * np.pi * X2['day'] / 30.5)

X1['month_sin'] = np.sin(2 * np.pi * X1['month'] / 12.0)
X1['month_cos'] = np.cos(2 * np.pi * X1['month'] / 12.0)

X2['month_sin'] = np.sin(2 * np.pi * X2['month'] / 12.0)
X2['month_cos'] = np.cos(2 * np.pi * X2['month'] / 12.0)

X1.wd = X1.wd.apply(wd_to_numeric)
X1['wd_sin'] = np.sin(2 * np.pi * X1['wd'] / 15.0)
X1['wd_cos'] = np.cos(2 * np.pi * X1['wd'] / 15.0)

X2.wd = X2.wd.apply(wd_to_numeric)
X2['wd_sin'] = np.sin(2 * np.pi * X2['wd'] / 15.0)
X2['wd_cos'] = np.cos(2 * np.pi * X2['wd'] / 15.0)

X1 = X1.values
Y1 = np.ravel(Y1.values)

# Split training data to create a training and a test set
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=1 / 3, shuffle=False)
n_samples, n_feats = X_train.shape

# Compute the mutual information between each feature and the target variable
mi_vec = mutual_info_regression(X_train, Y_train, n_neighbors=3)
most_mi = np.argsort(mi_vec)

fig = plt.figure(figsize=(10, 10), dpi=90)
feature_names = ['year', 'month', 'month_sin', 'month_cos', 'day', 'day_sin', 'day_cos', 'hour', 'hour_sin', 'hour_cos', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'wd', 'wd_sin', 'wd_cos', 'WSPM', 'station']
for i, ind in enumerate(most_mi):
    plt.subplot(n_feats // 3 + 1, 3, i + 1)
    plt.scatter(X_train[:, ind], Y_train, s=10)
    plt.title(f"{feature_names[ind]}: {mi_vec[ind]:.4}")
plt.tight_layout()
plt.show()

# Features selection : keep the 5 most informative
m = 5  # number of features kept

mi_best_indices = most_mi[-m:]
X_train_reduced = X_train[:, mi_best_indices]
X_test_reduced = X_test[:, mi_best_indices]
X1_reduced = X1[:, mi_best_indices]

# Standardisation scaling
scaler_X_train = StandardScaler()
X_train_scaled = scaler_X_train.fit_transform(X_train_reduced, Y_train)

scaler_Y_train = StandardScaler()
Y_train_scaled = np.ravel(scaler_Y_train.fit_transform(Y_train.reshape(-1, 1)))


# Model selection : hyperparameter tuning
alpha_grid = 10.0 ** -np.arange(1, 4)
parameters = {'alpha': alpha_grid, 'hidden_layer_sizes': (np.arange(6, 30, 6))}

nn = MLPRegressor(activation='tanh')
model = GridSearchCV(nn, parameters, scoring='neg_root_mean_squared_error', verbose=2, n_jobs=3)  # default 5-fold CV
model.fit(X_train_scaled, Y_train_scaled)

# Compute train error
pred_train = model.predict(X_train_scaled)
pred_train = scaler_Y_train.inverse_transform(pred_train)
error_train = sqrt(mean_squared_error(pred_train, Y_train))
print('Train RMSE: ' + str(error_train))

# Compute test error
pred_test = model.predict(X_test_reduced)
pred_test = scaler_Y_train.inverse_transform(pred_test)
error_test = sqrt(mean_squared_error(pred_test, Y_test))
print('Test RMSE: ' + str(error_test))

#%% Y2 prediction

X2 = X2.values
X2_reduced = X2[:, mi_best_indices]
X2_scaled = scaler_X_train.transform(X2_reduced)

Y2 = model.predict(X2_scaled)
Y2 = scaler_Y_train.inverse_transform(Y2)

Y2 = pd.DataFrame(Y2)
Y2.to_csv('Y2.csv')
