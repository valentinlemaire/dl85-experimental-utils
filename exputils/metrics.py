from collections.abc import Callable
import numpy as np
from scipy import integrate


def _sample_crps(y, cdf, imin, imax):
    lower_integrand = lambda x: np.square(cdf(x))
    upper_integrand = lambda x: np.square(1 - cdf(x))

    lower_integral, e, *_ = integrate.quad(lower_integrand, imin, y, limit=1000, full_output=True)
    
    if len(_) > 1:
        return None
    
    upper_integral, e, *_ = integrate.quad(upper_integrand, y, imax, limit=1000, full_output=True)
    if len(_) > 1:
        return None
    
    return lower_integral + upper_integral


def crps(
    x_true: np.ndarray,
    y_true: np.ndarray,
    cdf: Callable,
    imin:float = None,
    imax:float = None,
) -> float:
    """
    Computes the Continuous Ranked Probability score on a given dataset.

    Args:
        x_true (np.ndarray): the true x values, shape (n_samples, n_features)
        y_true (np.ndarray): the true y values, shape (n_samples,)
        cdf (Callable): the conditional cdf estimations, cdf(x, y) must give the conditional cdf of y given x
        imin (float, optional): The lower bound of the intergral. If None, min(y_true) - 5*std(y_true) is used. Defaults to None.
        imax (float, optional): The upper bound of the intergral. If None, max(y_true) + 5*std(y_true) is used. Defaults to None.

    Returns:
        float: the crps score
    """
    if imin is None:
        imin = np.min(y_true) - 5 * np.std(y_true)
    if imax is None:
        imax = np.max(y_true) + 5 * np.std(y_true)

    res = []
    for i, (x, y) in enumerate(zip(x_true, y_true)):
        if i % 100 == 0: 
            print(f"{i+1}/{len(x_true)}")
                  
        res.append(_sample_crps(
                y,
                lambda t: cdf(x, t),
                imin,
                imax, 
        ))
    res = np.array(res)
    
    return np.mean(res[res != None])


def log_likelihood(
    x_true: np.ndarray,
    y_true: np.ndarray,
    pdf: Callable,
    log_scale=False,
) -> float:
    """
    Computes the log-likelihood of a given dataset.

    Args:
        x_true (np.ndarray): the true x values, shape (n_samples, n_features)
        y_true (np.ndarray): the true y values, shape (n_samples, )
        pdf (Callable): the pdf estimations, pdf(x, y) must give the pdf of y given x
        log_scale (bool, optional): Indicates if the given estimated pdfs are in logarithmic scale. Defaults to False.

    Returns:
        float: the log-likelihood score
    """
    if log_scale:
        return -np.sum([pdf(x, y) for x, y in zip(x_true, y_true)])
    else:
        return -np.sum([np.log(pdf(x, y)) for x, y in zip(x_true, y_true)])


def mise(
    x_true: np.ndarray,
    true_pdf: Callable,
    pred_pdf: Callable,
    imin: float,
    imax: float,
    weights: np.ndarray = None,
) -> float:

    if weights is None:
        weights = np.zeros(len(x_true)) / len(x_true)
    if np.sum(weights) != 1:
        weights /= np.sum(weights)
    
    integrand = lambda x, y: (true_pdf(x, y) - pred_pdf(x, y)) ** 2
    integrals = np.array([integrate.quad(lambda y: integrand(x, y), imin, imax, limit=50)[0] for x in x_true])
    

    return np.dot(integrals, weights)


def quantile_error(y_true, y_pred, q):
    delta = y_pred - y_true
    delta[delta > 0] *= q
    delta[delta < 0] *= q-1 

    return np.mean(delta)

def mean_quantile_error(
    x_true: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray = None,
    quantile_function: Callable = None,
    quantiles: np.ndarray = np.linspace(0, 1, 100),
) -> float:
    """
    Computes the mean quantile error on a given dataset.

    Args:
        x_true (np.ndarray): the true x values, shape (n_samples, n_features)
        y_true (np.ndarray): the true y values, shape (n_samples, )
        y_pred (np.ndarray, optional): the predicted y values, must be of shape (n_samples, n_quantiles). Defaults to None. If None, quantile_function is used.
        quantile_function (Callable, optional): The estimated quantile function. quantile_function(x, q) must give the estimated quantile value for quantile q and sample x. Defaults to None. Only used if y_pred is None.
        quantiles (np.ndarray, optional): The quantiles for which y_pred was generated or on which quantile_function will be used. Defaults to np.linspace(0, 1, 100). 

    Raises:
        ValueError: if neither y_pred nor quantile_function are given.

    Returns:
        float: the mean quantile error
    """
    if y_pred is None:
        if quantile_function is None:
            raise ValueError("Either y_pred or quantile_function must be provided")
        y_pred = np.array([quantile_function(x, quantiles) for x in x_true])
    
    
    return np.mean([
        quantile_error(y_true, y_pred[:, i], q)
        for i, q in enumerate(quantiles)
    ])