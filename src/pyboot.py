from typing import Callable, Optional, Union
from inspect import isfunction
import numpy as np
import pandas as pd
from scipy.stats import norm


class Bootstrap:
    def __init__(self, b: int = 9_999, levels: Union[np.array, list] = [0.9, 0.95, 0.99],
                 method: list = ["norm", "basic", "perc", "bca"], sdfun: Optional[Callable] = None,
                 sdrep: int = 99, jacknife: Optional[Callable] = None, cl: bool = False, boot_dist: bool = False):
        self.B = b
        self.levels = levels
        self.method = method
        self.sdfun = sdfun
        self.sdrep = sdrep
        self.jacknife = jacknife
        self.cl = cl
        self.boot_dist = boot_dist
        self.nobs = 0
        self.nstat = 0
        self.nboot = b + 1
        self.nlevels = len(levels)
        self.alphas = Bootstrap._get_alphas(levels)
        self.results = {}

    @staticmethod
    def _get_alphas(levels: list) -> np.array:
        if isinstance(levels, list):
            levels = np.array(levels)
        return 1 - levels

    def _get_dimensions(self, t0: Union[int, float, np.array], varnames: Optional[list]) -> None:

        try:
            self.nstat = len(t0)
        except TypeError:
            self.nstat = 1

        self.dims = [self.nlevels, 2, self.nstat]
        self.dnames = [[f"{level*100}%" for level in self.levels],
                       ["lower", "upper"], varnames]

    def _make_jacknife(self, statistic, X):
        if "bca" in self.method:
            if not self.jackknife:
                return statistic  # returns callable
            else:
                return self.jackknife(X)
    def get_bootstrap(self, x, t0, statistic):
        bootx = np.random.choice(x, self.nobs * self.B).reshape((self.B, self.nobs))

        bootdist = np.zeros(self.nboot * self.nstat).reshape((self.nboot, self.nstat))
        bootdist[0,] = t0
        bootdist[1:self.nboot, ] = np.apply_along_axis(statistic, 1, bootx).T.reshape((self.nboot - 1, self.nstat))

        bootse = np.apply_along_axis(np.std, 0, bootdist)

        bootbias = np.apply_along_axis(np.mean, 0, bootdist) - t0

        return bootdist, bootse, bootbias

    def get_normal_interval(self, t0, bootse, bootbias):
        normal = np.zeros(len(self.dims) * 2).reshape((2, len(self.dims)))

        for i in range(self.nstat):
            # this will fail with nstat > 1, need to check this
            normal[0] = t0[i] - bootbias[i] - norm.ppf(1 - self.alphas / 2) * bootse[i]
            normal[1] = t0[i] - bootbias[i] - norm.ppf(self.alphas / 2) * bootse[i]

        return normal

    def get_perc_interval(self, bootdist):
        percent = np.zeros(len(self.dims) * 2).reshape((2, len(self.dims)))
        probs = list(self.alphas / 2) + list(1 - self.alphas / 2)

        for i in range(self.nstat):
            quant = np.nanquantile(bootdist[:, i], probs)
            percent[0] = quant[0:self.nlevels]
            percent[1] = quant[self.nlevels: len(probs)]

        return percent

    def get_basic_interval(self, t0, bootdist):
        basic = np.zeros(len(self.dims) * 2).reshape((2, len(self.dims)))
        probs = list(self.alphas / 2) + list(1 - self.alphas / 2)

        for i in range(self.nstat):
            quant = np.nanquantile(bootdist[:, i], probs)
            basic[0] = 2 * t0[i] - quant[self.nlevels:len(probs)]
            basic[1] = 2 * t0[i] - quant[0: self.nlevels]

        return basic

    def get_bca_interval(self, x, bootdist, jackknife, t0):
        bca = np.zeros(len(self.dims) * 2).reshape((2, len(self.dims)))
        z1 = norm.ppf(self.alphas / 2)
        z2 = norm.ppf(1 - self.alphas / 2)
        jackstat = np.zeros((self.nobs * self.nstat)).reshape((self.nobs, self.nstat))

        for i in range(len(x)):
            jackstat[i,] = jackknife(np.delete(x, i))

        jackmean = np.mean(jackstat, axis=0)
        z0 = np.zeros(self.nstat)
        acc = np.zeros(self.nstat)

        for i in range(self.nstat):
            z0[i] = norm.ppf(np.mean(bootdist[:, i] < t0, axis=0))
            acc[i] = np.sum((jackmean[i] - jackstat[:, i]) ** 3) / (
                        6 * np.sum((jackmean[i] - jackstat[:, i]) ** 2) ** (3 / 2))
            a = norm.cdf(z0[i] + (z0[i] + z1) / (1 - acc[i] * (z0[i] + z1)))
            b = norm.cdf(z0[i] + (z0[i] + z2) / (1 - acc[i] * (z0[i] + z2)))
            probs = list(a) + list(b)
            quant = np.nanquantile(bootdist[:, i], probs)
            bca[0] = quant[0:self.nlevels]
            bca[1] = quant[self.nlevels: len(probs)]

        return bca
    def estimate(self, X: Union[np.array, np.ndarray, pd.Series, pd.DataFrame], statistic: Callable,
                 **statistic_kwargs: dict):
        # placeholder to deal with varnames
        varnames = None
        self.nobs = len(X)
        t0 = statistic(X, **statistic_kwargs)
        self._get_dimensions(X, t0, varnames)
        bootdist, bootse, bootbias = self.get_bootstrap(X, t0, statistic)

        bootcov = np.cov(bootdist) if self.nstat > 1 else None
