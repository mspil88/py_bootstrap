from typing import Callable, Optional, Union
from inspect import isfunction
import numpy as np
import pandas as pd
from scipy.stats import norm


class Bootstrap:
    def __init__(self, b: int = 9_999, levels: Union[np.array, list] = [0.9, 0.95, 0.99],
                 method: list = ["norm", "basic", "perc", "bca"], sdfun: Optional[Callable] = None,
                 sdrep: int = 99, jackknife: Optional[Callable] = None, cl: bool = False, boot_dist: bool = False):
        self.B = b
        self.levels = levels
        self.method = method
        self.sdfun = sdfun
        self.sdrep = sdrep
        self.jackknife = jackknife
        self.cl = cl
        self.boot_dist = boot_dist
        self.nobs = 0
        self.nstat = 0
        self.nboot = b + 1
        self.nlevels = len(levels)
        self.alphas = Bootstrap._get_alphas(levels)
        self.probs = Bootstrap._get_probs(self.alphas)
        self.results = {}

    @staticmethod
    def _get_alphas(levels: list) -> np.array:
        if isinstance(levels, list):
            levels = np.array(levels)
        return 1 - levels

    @staticmethod
    def _get_probs(alphas: np.array) -> list:
        return list(alphas / 2) + list(1 - alphas / 2)

    def _get_dimensions(self, t0: Union[int, float, np.array], varnames: Optional[list]) -> None:

        try:
            self.nstat = len(t0)
        except TypeError:
            self.nstat = 1

        self.dims = [self.nlevels, 2, self.nstat]
        self.dnames = [[f"{level*100}%" for level in self.levels],
                       ["lower", "upper"], varnames]

    def _make_jackknife(self, statistic, X):
        if "bca" in self.method:
            if not self.jackknife:
                return statistic, False  # returns callable
            else:
                return self.jackknife(X), True
    def get_bootstrap(self, x, t0, statistic, **statistic_kwargs):
        bootx = np.random.choice(x, self.nobs * self.B).reshape((self.B, self.nobs))

        bootdist = np.zeros(self.nboot * self.nstat).reshape((self.nboot, self.nstat))
        bootdist[0,] = t0

        bootdist[1:self.nboot, ] = np.apply_along_axis(statistic, 1, bootx, **statistic_kwargs).reshape((self.nboot - 1, self.nstat))

        bootse = np.apply_along_axis(np.std, 0, bootdist)

        bootbias = np.apply_along_axis(np.mean, 0, bootdist) - t0

        return bootdist, bootse, bootbias

    def create_bootstrap_container(self):
        return np.zeros(self.nstat*len(self.dims) * 2).reshape((self.nstat, 2, len(self.dims)))

    def get_normal_interval(self, t0, bootse, bootbias):

        normal = self.create_bootstrap_container()
        for i in range(self.nstat):
            normal[i, 0] = t0[i] - bootbias[i] - norm.ppf(1 - self.alphas / 2) * bootse[i]
            normal[i, 1] = t0[i] - bootbias[i] - norm.ppf(self.alphas / 2) * bootse[i]

        return normal

    def get_perc_interval(self, bootdist):
        percent = self.create_bootstrap_container()

        for i in range(self.nstat):
            quant = np.nanquantile(bootdist[:, i], self.probs)
            percent[i, 0] = quant[0:self.nlevels]
            percent[i, 1] = quant[self.nlevels: len(self.probs)]

        return percent

    def get_basic_interval(self, t0, bootdist):
        basic = self.create_bootstrap_container()

        for i in range(self.nstat):
            quant = np.nanquantile(bootdist[:, i], self.probs)
            basic[i, 0] = 2 * t0[i] - quant[self.nlevels:len(self.probs)]
            basic[i, 1] = 2 * t0[i] - quant[0: self.nlevels]

        return basic

    def get_bca_interval(self, X, bootdist, t0, statistic: Callable, statistic_kwargs: dict):
        bca = self.create_bootstrap_container()
        z1 = norm.ppf(self.alphas / 2)
        z2 = norm.ppf(1 - self.alphas / 2)
        jackstat = np.zeros((self.nobs * self.nstat)).reshape((self.nobs, self.nstat))

        jackknife, custom_jackknife = self._make_jackknife(statistic, X)

        for i in range(len(X)):
            if not custom_jackknife:
                jackstat[i,] = jackknife(np.delete(X, i), **statistic_kwargs)
            else:
                jackstat[i,] = jackknife(np.delete(X, i), **statistic_kwargs)

        jackmean = np.mean(jackstat, axis=0)
        z0 = np.zeros(self.nstat)
        acc = np.zeros(self.nstat)

        for i in range(self.nstat):
            z0[i] = norm.ppf(np.mean(bootdist[:, i] < t0[i], axis=0))
            acc[i] = np.sum((jackmean[i] - jackstat[:, i]) ** 3) / (
                        6 * np.sum((jackmean[i] - jackstat[:, i]) ** 2) ** (3 / 2))
            a = norm.cdf(z0[i] + (z0[i] + z1) / (1 - acc[i] * (z0[i] + z1)))
            b = norm.cdf(z0[i] + (z0[i] + z2) / (1 - acc[i] * (z0[i] + z2)))
            probs = list(a) + list(b)
            quant = np.nanquantile(bootdist[:, i], probs)
            bca[i, 0] = quant[0:self.nlevels]
            bca[i, 1] = quant[self.nlevels: len(probs)]

        return bca

    def bootstrap_method_results(self, results: list, varnames: list = None):
        if results is None:
            return None

        method_result = {}
        n_results = len(results)

        if varnames is None:
            varnames = [f'estimate{i}' for i in range(n_results)]

        for varname, result in zip(varnames, results):
            lower = result[0]
            upper = result[1]

            lower_res = {}
            upper_res = {}

            for low, up, prob in zip(lower, upper, self.probs):
                lower_res[prob] = low
                upper_res[prob] = up

            method_result[varname] = {"lower": lower_res, "upper": upper_res }

        return method_result

    def estimate(self, X: Union[np.array, np.ndarray, pd.Series, pd.DataFrame], statistic: Callable,
                 **statistic_kwargs: dict):
        # placeholder to deal with varnames
        varnames = None
        self.nobs = len(X)
        t0 = statistic(X, **statistic_kwargs)
        t0 = np.array([t0]) if isinstance(t0, np.float64) else t0
        self._get_dimensions(t0, varnames)
        bootdist, bootse, bootbias = self.get_bootstrap(X, t0, statistic, **statistic_kwargs)

        bootcov = np.cov(bootdist) if self.nstat > 1 else None

        normal = self.get_normal_interval(t0, bootse, bootbias) if "norm" in self.method else None
        percent = self.get_perc_interval(bootdist) if "perc" in self.method else None
        basic = self.get_basic_interval(t0, bootdist) if "basic" in self.method else None
        bca = self.get_bca_interval(X, bootdist, t0, statistic, statistic_kwargs) if "bca" in self.method else None

        return normal