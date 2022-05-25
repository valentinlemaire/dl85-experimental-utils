from sklearn.exceptions import NotFittedError
from dl85 import DL85Regressor, DL85Predictor
from sklearn.base import RegressorMixin
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import resource
from memory_profiler import memory_usage
from time import sleep

class MemoryMonitor:
    def __init__(self, interval=1e-2):
        self.keep_measuring = True
        self.interval = interval

    def measure_usage(self):
        usage = []
        while self.keep_measuring:
            usage.append(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            sleep(self.interval)

        return usage

def fit_model(model, X, y):
    model.fit(X, y)
    del model.leaf_value_function

def memory_measurement(model, X, y):

    return memory_usage((fit_model, (model, X, y)), interval=1e-2)
    

class DL85QuantileRegressor(RegressorMixin):
    def __init__(
        self,
        max_depth=1,
        min_sup=1,
        max_error=0,
        quantile_estimation="linear",
        quantiles=[0.5],
        stop_after_better=False,
        time_limit=0,
        verbose=False,
        desc=False,
        asc=False,
        repeat_sort=False,
        print_output=False,
        n_jobs=1,
    ):
        """
        This class simulates the DL85QuantileRegressor class from the dl85 package by training one model for each quantile.

        See dl85 documentation for more information.
        """
        self.max_depth = max_depth
        self.min_sup = min_sup
        self.max_error = max_error
        self.stop_after_better = stop_after_better
        self.time_limit = time_limit
        self.verbose = verbose
        self.desc = desc
        self.asc = asc
        self.repeat_sort = repeat_sort
        self.print_output = print_output
        self.quantiles = quantiles
        self.n_jobs = n_jobs

        self.regressors = {
            q: DL85Regressor(
                max_depth=max_depth,
                min_sup=min_sup,
                max_error=max_error,
                stop_after_better=stop_after_better,
                time_limit=time_limit,
                verbose=verbose,
                desc=desc,
                asc=asc,
                repeat_sort=repeat_sort,
                backup_error="quantile",
                quantile_value=q,
                print_output=print_output,
                quantile_estimation=quantile_estimation,
            )
            for q in self.quantiles
        }

        self.memory_usage = []

    def fit(self, X, y=None):
        for q, r in self.regressors.items():
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                p = executor.submit(memory_measurement, r, X, y)

                usage = p.result()
                self.memory_usage.append(usage)
        return self

    def predict(self, X, q=0.5):
        if q not in self.quantiles:
            raise NotFittedError(
                f"This predictor has not been fitted on the {q} quantile."
            )

        return self.regressors[q].predict(X)

    def save(self, filename):
        data_dict = {
            "params": {
                "max_depth": self.max_depth,
                "min_sup": self.min_sup,
                "max_error": self.max_error,
                "quantiles": list(self.quantiles),
                "stop_after_better": self.stop_after_better,
                "time_limit": self.time_limit,
                "verbose": self.verbose,
                "desc": self.desc,
                "asc": self.asc,
                "repeat_sort": self.repeat_sort,
                "print_output": self.print_output,
                "n_jobs": self.n_jobs,
            },
            "regressors": {
                q: {
                    "params": {
                        "max_depth": r.max_depth,
                        "min_sup": r.min_sup,
                        "max_error": r.max_error,
                        "stop_after_better": r.stop_after_better,
                        "time_limit": r.time_limit,
                        "verbose": r.verbose,
                        "desc": r.desc,
                        "asc": r.asc,
                        "repeat_sort": r.repeat_sort,
                        "backup_error": r.backup_error,
                        "quantile_value": r.quantile_value,
                        "print_output": r.print_output,
                    },
                    "extra": {
                        "tree_": r.tree_,
                        "is_fitted_": r.is_fitted_,
                    },
                }
                for q, r in self.regressors.items()
            },
        }

        # print(json.dumps(data_dict))

        with open(filename, "w") as outfile:
            json.dump(data_dict, outfile)

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, "r") as infile:
            data = json.load(infile)

        quantile_regressor = DL85QuantileRegressor(**data["params"])

        regressors = {}
        for q, reg_data in data["regressors"].items():
            regressors[float(q)] = DL85Regressor(**reg_data["params"])
            regressors[float(q)].is_fitted_ = reg_data["extra"]["is_fitted_"]
            regressors[float(q)].tree_ = reg_data["extra"]["tree_"]
            regressors[float(q)].criterion = "quantile"
            # regressors[float(q)].tree_.n_outputs = 1

        quantile_regressor.regressors = regressors
        return quantile_regressor

    @classmethod
    def from_json_str(cls, str):
        data = json.loads(str)

        quantile_regressor = DL85QuantileRegressor(**data["params"])

        regressors = {}
        for q, reg_data in data["regressors"].items():
            regressors[float(q)] = DL85Regressor(**reg_data["params"])
            regressors[float(q)].is_fitted_ = reg_data["extra"]["is_fitted_"]
            regressors[float(q)].tree_ = reg_data["extra"]["tree_"]

        quantile_regressor.regressors = regressors
        return quantile_regressor


