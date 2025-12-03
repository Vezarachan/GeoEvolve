# EVOLVE-BLOCK-START
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from typing import Union


def exponential_variogram(h: Union[float, NDArray], nugget: float, sill: float, range_: float) -> Union[float, NDArray]:
    return nugget + sill * (1 - np.exp(-h * range_))

def linear_variogram(h: Union[float, NDArray], nugget: float, sill: float, range_: float) -> Union[float, NDArray]:
    return nugget + sill * np.minimum(h / range_, 1.0)

def gaussian_variogram(h: Union[float, NDArray], nugget: float, sill: float, range_: float) -> Union[float, NDArray]:
    return nugget + sill * (1 - np.exp(-(h / range_) ** 2))

def empirical_variogram(coords, values, n_lags=20, max_range=None):
    dists = cdist(coords, coords)
    diffs = 0.5 * (values[:, None] - values[None, :])**2

    if max_range is None:
        max_range = np.max(dists) * 0.5

    lag_bins = np.linspace(0, max_range, n_lags + 1)
    gamma = np.zeros(n_lags)
    lags = np.zeros(n_lags)

    for i in range(n_lags):
        mask = (dists >= lag_bins[i]) & (dists < lag_bins[i+1])
        if np.any(mask):
            gamma[i] = np.mean(diffs[mask])
            lags[i] = np.mean(dists[mask])

    return lags, gamma

def variogram_l1_loss(params, lags, empirical_gamma):
    nugget, sill, rng = params
    if sill < nugget or rng <= 0:
        return np.inf
    model_gamma = exponential_variogram(lags, nugget, sill, rng)

    abs_error = np.abs(empirical_gamma - model_gamma)
    return np.mean(abs_error)

def fit_variogram_model_l1(lags, gamma):
    # Smart initial guess
    nugget0 = min(gamma)
    sill0 = max(gamma)
    range0 = max(lags)
    initial_guess = (nugget0, sill0, range0)

    bounds = [(0, None), (0, None), (1e-6, None)]

    result = minimize(
        variogram_l1_loss,
        initial_guess,
        args=(lags, gamma),
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        return result.x  # nugget, sill, range
    else:
        raise RuntimeError("L1 variogram fitting failed: " + result.message)

def ordinary_kriging(xy_known, values_known, xy_unknown, variogram_func, variogram_params):
    n = len(xy_known)
    nugget, sill, rng = variogram_params

    # Distance matrices
    dists_known = cdist(xy_known, xy_known)
    dists_unknown = cdist(xy_unknown, xy_known)

    # Variogram values (covariance matrix)
    gamma_matrix = variogram_func(dists_known, nugget, sill, rng)
    gamma_vector = variogram_func(dists_unknown, nugget, sill, rng)

    # Build Kriging system (with Lagrange multiplier)
    K = np.zeros((n+1, n+1))
    K[:n, :n] = gamma_matrix
    K[:n, -1] = 1
    K[-1, :n] = 1
    K[n , n] = 0

    predictions = []

    for gv in gamma_vector:
        rhs = np.append(gv, 1)
        weights = np.linalg.solve(K, rhs)
        prediction = np.sum(weights[:-1] * values_known)
        predictions.append(prediction)

    return np.array(predictions)

# EVOLVE-BLOCK-END

def run_kfold(k=5, mineral_type='Cu'):
    from sklearn.model_selection import KFold
    from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
    data = pd.read_csv(f'/Users/louxiayin/PycharmProjects/GeoEvolve/examples/kriging/data/data_{mineral_type}.csv')
    rmse_list = np.zeros(k)

    kfold = KFold(n_splits=k, shuffle=True)
    for i, (train_index, test_index) in enumerate(kfold.split(data)):
        data_train = data.iloc[train_index, :]
        data_test = data.iloc[test_index, :]
        loc_known = data_train[['DLONG', 'DLAT']].values
        z_known = data_train[[f'{mineral_type}_ppm']].values.flatten()
        z_known = np.log(z_known + 1e-8)  # add small value to avoid log(0)
        loc_unknown = data_test[['DLONG', 'DLAT']].values
        z_unknown = data_test[[f'{mineral_type}_ppm']].values.flatten()
        z_unknown = np.log(z_unknown + 1e-8)
        lags, gamma = empirical_variogram(loc_known, z_known)
        variogram_params = fit_variogram_model_l1(lags, gamma)
        z_pred = ordinary_kriging(loc_known, z_known, loc_unknown, exponential_variogram, variogram_params)
        z_pred[z_pred < 0] = 0
        rmse = root_mean_squared_error(z_unknown, z_pred)
        rmse_list[i] = rmse
    return rmse_list.mean()

def run_ok():
    mean_rmse_list = []
    for mineral_type in ['Cu', 'Pb', 'Zn']:
        mean_rmse = run_kfold(5, mineral_type)
        mean_rmse_list.append(mean_rmse)
    return np.mean(mean_rmse_list), mean_rmse_list


def run_kfold_evaluation(k=5, mineral_type='Cu'):
    from sklearn.model_selection import KFold
    from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
    data = pd.read_csv(f'/Users/louxiayin/PycharmProjects/GeoEvolve/examples/kriging/data/data_{mineral_type}.csv')
    rmse_list = np.zeros(k)
    mae_list = np.zeros(k)
    r2_list = np.zeros(k)

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kfold.split(data)):
        data_train = data.iloc[train_index, :]
        data_test = data.iloc[test_index, :]
        loc_known = data_train[['DLONG', 'DLAT']].values
        z_known = data_train[[f'{mineral_type}_ppm']].values.flatten()
        # Log transform with offset for skew, but use median offset for robustness
        z_known = np.log(z_known + 1e-8)
        loc_unknown = data_test[['DLONG', 'DLAT']].values
        z_unknown = data_test[[f'{mineral_type}_ppm']].values.flatten()
        z_unknown = np.log(z_unknown + 1e-8)
        lags, gamma = empirical_variogram(loc_known, z_known)
        variogram_params = fit_variogram_model_l1(lags, gamma)
        z_pred = ordinary_kriging(loc_known, z_known, loc_unknown, exponential_variogram, variogram_params)
        z_pred[z_pred < 0] = 0
        rmse = root_mean_squared_error(z_unknown, z_pred)
        mae = mean_absolute_error(z_unknown, z_pred)
        r2 = r2_score(z_unknown, z_pred)
        rmse_list[i] = rmse
        mae_list[i] = mae
        r2_list[i] = r2
    # print(f'Avg RMSE: {np.mean(rmse_list):.4f}')
    # print(f'Avg R2: {np.mean(r2_list):.4f}')
    return np.mean(rmse_list), np.mean(mae_list), np.mean(r2_list)


if __name__ == '__main__':
    # result = run_ok()
    # print(result[0])
    # print(result[1])
    # print(result)

    print('Initial OK')
    df = {}
    for mineral_type in ['Cu', 'Pb', 'Zn']:
        rmse_list = []
        mae_list = []
        r2_list = []

        for k in range(5, 11):
            mean_rmse, mean_mae, mean_r2 = run_kfold_evaluation(k, mineral_type)
            rmse_list.append(mean_rmse)
            mae_list.append(mean_mae)
            r2_list.append(mean_r2)
        print(f'{mineral_type}')
        print(
            f'Mean RMSE: {np.mean(rmse_list):.4f}, Mean MAE: {np.mean(mae_list):.4f}, Mean R2: {np.mean(r2_list):.4f}')
        df[mineral_type] = {'mean_rmse': np.mean(rmse_list), 'mean_mae': np.mean(mae_list), 'mean_r2': np.mean(r2_list)}
    df = pd.DataFrame(df)
    df.to_csv('results_with_geo_know.csv')
