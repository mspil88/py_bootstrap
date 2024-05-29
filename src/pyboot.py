from typing import Callable, Optional, Union
from inspect import isfunction
import numpy as np
import pandas as pd


class Bootstrap:
    def __init__(self, b: int = 9_999, level: Union[np.array, list] = [0.9, 0.95, 0.99],
                 method: list = ["norm", "basic", "perc", "bca"], sdfun: Optional[Callable] = None,
                 sdrep: int = 99, jacknife: Optional[Callable] = None, cl: bool = False, boot_dist: bool = False):
        self.B = b
        self.level = level
        self.method= method
        self.sdfun = sdfun
        self.sdrep = sdrep
        self.jacknife = jacknife
        self.cl = cl
        self.boot_dist = boot_dist
        self.nobs = 0
        self.nstat = 0
        self.nboot = b + 1
        self.nlevel = len(level)
        self.alphas = Bootstrap._get_alphas(level)
        self.results = {}

    @staticmethod
    def _get_alphas(levels):
        if isinstance(levels, list):
            levels = np.array(levels)
        return 1 - levels

    def get_bootstrap(self, x, t0, statistic):
        bootx = np.random.choice(x, self.nobs * self.B).reshape((self.B, self.nobs))

        bootdist = np.zeros(self.nboot * self.nstat).reshape((self.nboot, self.nstat))
        bootdist[0,] = t0
        bootdist[1:self.nboot, ] = np.apply_along_axis(statistic, 1, bootx).T.reshape((self.nboot - 1, self.nstat))

        bootse = np.apply_along_axis(np.std, 0, bootdist)

        bootbias = np.apply_along_axis(np.mean, 0, bootdist) - t0

        return bootdist, bootse, bootbias

    def estimate(self, X: Union[np.array, np.ndarray, pd.Series, pd.DataFrame], statistic: Callable,
                 **statistic_kwargs: dict):
        self.nobs = len(X)
        t0 = statistic(X, **statistic_kwargs)
        bootdist, bootse, bootbias = self.get_bootstrap(X, t0, statistic)

        bootcov = np.cov(bootdist) if self.nstat > 1 else None

