from codecs import utf_16_be_encode
from sklearn.exceptions import NotFittedError
from dl85 import DL85QuantileRegressor
import numpy as np
from math import floor, ceil
from scipy.stats import gaussian_kde
import json
from skgarden import RandomForestQuantileRegressor

DEFAULT_N_SAMPLES = 1000
DEFAULT_N_QUANTILES = 100


class DistributionRegressor:
    def __init__(
        self,
        regressor,
        kernel_bandwidth: str = "scott",
        n_quantiles: int = DEFAULT_N_QUANTILES,
    ) -> None:
        """
        A wrapper for dl85's DL85QuantileRegressor or skgarden's RandomForestQuantileRegressor that provides a distribution estimator on top of a quantile regressor.

        Args:
            regressor (DL85QuantileRegressor | RandomForestQuantileRegressor): The quantile regressor to use.
            kernel_bandwidth (str, optional): the badwidth to use for the kernel density estimation, can also be a method string in {"scott", "silverman"}. Defaults to "scott". See scipy.stats.gaussian_kde for more information.
            n_quantiles (int, optional): number of quantiles to predict when using RandomForestQuantileRegressor. Defaults to 100.
        """

        self.regressor = regressor
        self.kernel_bandwidth = kernel_bandwidth
        self.kde: gaussian_kde = None
        self.n_quantiles = n_quantiles

    def fit(self, X):
        if isinstance(self.regressor, DL85QuantileRegressor):
            y = np.array(self.regressor.predict([X])).ravel()
        elif isinstance(self.regressor, RandomForestQuantileRegressor):
            y = []
            for q in np.linspace(0, 1, self.n_quantiles):
                y.append(self.regressor.predict([X], q*100)[0])
            y = np.array(y)
        self.kde = gaussian_kde(y, bw_method=self.kernel_bandwidth)

        return self

    def predict(self, y):
        if not self.kde:
            raise NotFittedError("DistributionRegressor is not fitted yet.")
        return self.kde.pdf(y)

    def log_predict(self, y):
        if not self.kde:
            raise NotFittedError("DistributionRegressor is not fitted yet.")

        return self.kde.logpdf(y)

    def integrate_1d(self, lb, ub):
        if not self.kde:
            raise NotFittedError("DistributionRegressor is not fitted yet.")

        return self.kde.integrate_box_1d(lb, ub)
