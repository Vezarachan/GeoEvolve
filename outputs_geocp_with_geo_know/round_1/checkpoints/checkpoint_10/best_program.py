# EVOLVE-BLOCK-START
import os.path
from typing import Callable, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray


def gaussian_kernel(d: NDArray) -> NDArray:
    """
    Gaussian distance decay function
    :param d: distances from test samples to calibration samples
    :return: list of weights for calibration samples
    """
    return np.exp(-0.5 * d ** 2)


def kernel_smoothing(z_test: NDArray, z_calib: NDArray, bandwidth: float) -> NDArray:
    """
    Kernel smoothing function
    :param z_test: the coordinates of test samples
    :param z_calib: the coordinates of calibration samples
    :param bandwidth: distance decay parameter
    :return: list of normalized weights for calibration samples
    """
    z_test_norm = np.sum(z_test ** 2, axis=1).reshape(-1, 1)
    z_calib_norm = np.sum(z_calib ** 2, axis=1).reshape(1, -1)
    distances = np.sqrt(z_test_norm + z_calib_norm - 2 * np.dot(z_test, z_calib.T))
    weights = gaussian_kernel(distances / bandwidth)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    # Avoid division by zero
    weights_sum[weights_sum == 0] = 1
    weights = weights / weights_sum
    return weights


def weighted_quantile(scores: NDArray, weights: NDArray, q: float):
    """
    Calculate weighted quantile
    :param scores: nonconformity scores
    :param q: quantile level
    :param weights: geographic weights (normalized)
    :return: weighted quantile at (1-alpha) miscoverage level
    """
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)

    N_test, N_calib = weights.shape
    scores = scores.reshape(1, -1)
    scores_repeated = np.repeat(scores, N_test, axis=0)
    sorted_idx = np.argsort(scores_repeated, axis=1)
    scores_sorted = np.take_along_axis(scores_repeated, sorted_idx, axis=1)
    weights_sorted = np.take_along_axis(weights, sorted_idx, axis=1)

    cumulative_weights = np.cumsum(weights_sorted, axis=1)
    idx = np.argmax(cumulative_weights >= q, axis=1)
    quantiles = scores_sorted[np.arange(N_test), idx]
    return quantiles


class GeoConformalBase:
    def __init__(self, predict_f: Callable, x_calib: NDArray, y_calib: NDArray, coord_calib: NDArray,
                 bandwidth: Union[float, int], miscoverage_level: float = 0.1):
        self.predict_f = predict_f
        self.bandwidth = bandwidth
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.coord_calib = coord_calib
        self.miscoverage_level = miscoverage_level

    def geo_conformalized(self, x_test: NDArray, y_test: NDArray, coord_test: NDArray):
        raise NotImplementedError


class GeoConformalRegressor(GeoConformalBase):
    """
    Parameters
    ----------
    predict_f: spatial prediction function (regression or interpolation)
    """

    def __init__(self, predict_f: Callable, x_calib: NDArray, y_calib: NDArray, coord_calib: NDArray,
                 bandwidth: Union[float, int], miscoverage_level: float = 0.1):
        super().__init__(predict_f, x_calib, y_calib, coord_calib, bandwidth, miscoverage_level)

    def geo_conformalize(self,
                         x_test: NDArray,
                         y_test: NDArray,
                         coord_test: NDArray):
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.abs(y_calib_pred - self.y_calib)
        N = nonconformity_scores.shape[0]
        # Use standard conformal quantile
        q_level = 1 - self.miscoverage_level
        weights = kernel_smoothing(coord_test, self.coord_calib, self.bandwidth)
        geo_uncertainty = weighted_quantile(nonconformity_scores, weights, q_level)
        y_test_pred = self.predict_f(x_test)
        upper_bound = y_test_pred + geo_uncertainty
        lower_bound = y_test_pred - geo_uncertainty
        return geo_uncertainty, upper_bound, lower_bound, y_test


# EVOLVE-BLOCK-END

def run_geocp():
    train = pd.read_csv(f'/Users/louxiayin/PycharmProjects/GeoEvolve/examples/geocp/data/train.csv')
    calib = pd.read_csv(f'/Users/louxiayin/PycharmProjects/GeoEvolve/examples/geocp/data/calib.csv')
    test = pd.read_csv(f'/Users/louxiayin/PycharmProjects/GeoEvolve/examples/geocp/data/test.csv')
    variables = ['bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'condition', 'waterfront', 'view', 'age', 'UTM_X',
                 'UTM_Y']
    X_train, y_train = train[variables], train['price']
    X_calib, y_calib, loc_calib = calib[variables], calib['price'], calib[['lat', 'lon']]
    X_test, y_test, loc_test = test[variables], test['price'], test[['lat', 'lon']]
    model = xgb.XGBRegressor(n_estimators=500, max_depth=3, min_child_weight=1.0, colsample_bytree=1.0).fit(
        X_train.values, y_train.values)
    geocp_regresser = GeoConformalRegressor(predict_f=model.predict, x_calib=X_calib.values, y_calib=y_calib.values,
                                            coord_calib=loc_calib.values, bandwidth=0.15, miscoverage_level=0.1)
    geo_uncertainty, upper_bound, lower_bound, y_test = geocp_regresser.geo_conformalize(X_test.values, y_test.values,
                                                                                         loc_test.values)
    return geo_uncertainty, upper_bound, lower_bound, y_test


def interval_score(y_true, lower_bound, upper_bound, alpha=0.1, epsilon=1e-6):
    width = np.maximum(upper_bound - lower_bound, epsilon)
    below = (lower_bound - y_true) * (y_true < lower_bound)
    above = (y_true - upper_bound) * (y_true > upper_bound)
    score = width + (2 / alpha) * (below + above)
    score = np.where(np.isnan(score), 0.0, score)
    return np.mean(score)


def run_k_times(k=5):
    from sklearn.model_selection import train_test_split
    base_path = '/Users/louxiayin/PycharmProjects/GeoEvolve/examples/geocp/data'
    data = pd.read_csv(os.path.join(base_path, 'seattle_sample_3k.csv'))
    data = gpd.GeoDataFrame(data, crs="EPSG:32610", geometry=gpd.points_from_xy(x=data.UTM_X, y=data.UTM_Y))
    data = data.to_crs(4326)
    data['lon'] = data['geometry'].get_coordinates()['x']
    data['lat'] = data['geometry'].get_coordinates()['y']
    data['price'] = np.power(10, data['log_price']) / 10000
    X = data[
        ['bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'condition', 'waterfront', 'view', 'age', 'UTM_X', 'UTM_Y']]
    y = data['price']
    loc = data[['lat', 'lon']]
    scores = np.zeros(k)
    for i in range(k):
        X_train, X_temp, y_train, y_temp, _, loc_temp = train_test_split(X, y, loc, train_size=0.8, random_state=42)
        X_calib, X_test, y_calib, y_test, loc_calib, loc_test = train_test_split(X_temp, y_temp, loc_temp,
                                                                                 train_size=0.5, random_state=42)
        model = xgb.XGBRegressor(n_estimators=600, max_depth=5, min_child_weight=0.8, colsample_bytree=0.7,
                                 subsample=0.8,
                                 learning_rate=0.05, random_state=42).fit(
            X_train.values, y_train.values)
        geocp_regresser = GeoConformalRegressor(
            predict_f=model.predict,
            x_calib=X_calib.values, y_calib=y_calib.values,
            coord_calib=loc_calib.values, bandwidth=0.15, miscoverage_level=0.1
        )
        geo_uncertainty, upper_bound, lower_bound, y_test = geocp_regresser.geo_conformalize(
            X_test.values, y_test.values, loc_test.values
        )
        scores[i] = interval_score(y_test, lower_bound, upper_bound, alpha=0.1)
    return np.mean(scores)


if __name__ == "__main__":
    # geo_uncertainty, upper_bound, lower_bound, y_test = run_geocp()
    # print(geo_uncertainty, upper_bound, lower_bound, y_test)
    mean_interval_score = run_k_times(k=20)
    print(f'Initial Code')
    print(f'Mean Interval Score: {mean_interval_score}')
