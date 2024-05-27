from typing import Callable, Optional, Union
from inspect import isfunction
import numpy as np


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
        self.alphas = Bootstrap.get_alphas(level)
        self.results = {}
