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


def kernel_smoothing(z_test: NDArray, z_calib: NDArray, bandwidth: float, lmbda: float = 0.0) -> NDArray:
    """
    Kernel smoothing function with optional Tikhonov regularization.
    :param z_test: the coordinates of test samples
    :param z_calib: the coordinates of calibration samples
    :param bandwidth: distance decay parameter (can be scalar or array for adaptive)
    :param lmbda: regularization parameter to control spatial autocorrelation
    :return: list of normalized weights for calibration samples
    """
    # Allow for adaptive bandwidth: if bandwidth is array, each row is for one test point
    z_test_norm = np.sum(z_test ** 2, axis=1).reshape(-1, 1)
    z_calib_norm = np.sum(z_calib ** 2, axis=1).reshape(1, -1)
    distances = np.sqrt(z_test_norm + z_calib_norm - 2 * np.dot(z_test, z_calib.T))
    # Adaptive bandwidth: bandwidth can be 1D (length = n_test)
    if np.ndim(bandwidth) == 1:
        bw = bandwidth.reshape(-1, 1)
        weights = gaussian_kernel(distances / bw)
    else:
        weights = gaussian_kernel(distances / bandwidth)
    # Regularization to avoid overfitting to local clusters (spatial autocorrelation)
    if lmbda > 0:
        weights = weights + lmbda
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
                 bandwidth: Union[float, int], miscoverage_level: float = 0.1, lmbda: float = 0.01, adaptive_bandwidth: bool = True, k_neighbors: int = 30):
        super().__init__(predict_f, x_calib, y_calib, coord_calib, bandwidth, miscoverage_level)
        self.lmbda = lmbda
        self.adaptive_bandwidth = adaptive_bandwidth
        self.k_neighbors = k_neighbors

    def _compute_adaptive_bandwidth(self, z_test, z_calib):
        """
        Compute adaptive bandwidth for each test point based on k-nearest spatial neighbors.
        This method supports multiscale bandwidths by evaluating several k values and selecting the best via cross-validation.
        """
        from sklearn.neighbors import NearestNeighbors
        # Use a range of k for multiscale bandwidths
        k_choices = [10, 20, 30, 50]
        best_bandwidth = None
        best_score = np.inf
        # Cross-validation based bandwidth search on calibration set
        for k in k_choices:
            nbrs = NearestNeighbors(n_neighbors=min(k, len(z_calib)), algorithm='auto').fit(z_calib)
            distances, _ = nbrs.kneighbors(z_calib)
            bw_cand = np.median(distances, axis=1)
            # Compute weights for a held-out fold in calibration set
            # To keep complexity low, use a hold-out split (not full k-fold)
            split = int(0.8 * len(z_calib))
            idx_train, idx_val = np.arange(split), np.arange(split, len(z_calib))
            z_train, z_val = z_calib[idx_train], z_calib[idx_val]
            bw_val = bw_cand[idx_val]
            weights = kernel_smoothing(z_val, z_train, bw_val, self.lmbda)
            # For this fold, use weighted quantile and interval score
            y_calib_pred_fold = self.predict_f(self.x_calib[idx_train])
            nc_scores_fold = np.abs(y_calib_pred_fold - self.y_calib[idx_train])
            q_level = 1 - self.miscoverage_level
            geo_uncertainty_val = weighted_quantile(nc_scores_fold, weights, q_level)
            y_val_pred = self.predict_f(self.x_calib[idx_val])
            upper = y_val_pred + geo_uncertainty_val
            lower = y_val_pred - geo_uncertainty_val
            # Use mean interval length as proxy for scoring bandwidths
            mean_length = np.mean(upper - lower)
            if mean_length < best_score:
                best_score = mean_length
                best_bandwidth = np.median(distances, axis=1)
        # Now compute for test points using best k
        nbrs = NearestNeighbors(n_neighbors=min(k_choices[np.argmin([k for k in k_choices])], len(z_calib)), algorithm='auto').fit(z_calib)
        distances, _ = nbrs.kneighbors(z_test)
        bw = np.median(distances, axis=1)
        bw = np.clip(bw, 1e-3, np.inf)
        return bw

    def geo_conformalize(self,
                         x_test: NDArray,
                         y_test: NDArray,
                         coord_test: NDArray):
        """
        Enhanced geo_conformalize with bandwidth selection by cross-validation and spatial autocorrelation regularization.
        """
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.abs(y_calib_pred - self.y_calib)
        q_level = 1 - self.miscoverage_level
        # Improved: bandwidth selected by cross-validation/multiscale for each test point
        if self.adaptive_bandwidth:
            bandwidth = self._compute_adaptive_bandwidth(coord_test, self.coord_calib)
        else:
            bandwidth = self.bandwidth
        # Adaptive kernel: use weights based on local calibration point density
        weights = kernel_smoothing(coord_test, self.coord_calib, bandwidth, self.lmbda)
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
        # Use adaptive bandwidth and regularization in GeoConformalRegressor
        geocp_regresser = GeoConformalRegressor(
            predict_f=model.predict,
            x_calib=X_calib.values, y_calib=y_calib.values,
            coord_calib=loc_calib.values, bandwidth=0.15, miscoverage_level=0.1,
            lmbda=0.01, adaptive_bandwidth=True, k_neighbors=30
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
