"""
Вероятностные распределения, используемые в проекте.
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from scipy.stats import norm
import numpy as np
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class SplitNormal:
    """
    Сплит-нормальное (split-normal) распределение, параметризованное медианой
    и расстояниями до 0.16 и 0.84 квантилей.

    Parameters
    ----------
    median : float
        Медиана распределения.
    lower_err : float > 0
        Расстояние от медианы до 0.16 квантиля.
    upper_err : float > 0
        Расстояние от медианы до 0.84 квантиля.
    """

    def __init__(self, median=0.0, lower_err=1.0, upper_err=1.0):
        if lower_err <= 0 or upper_err <= 0:
            raise ValueError("Errors must be strictly positive.")
        self.median = median
        a = norm.ppf(0.84)          # ≈ 0.994457883
        self.sigma_left = lower_err / a
        self.sigma_right = upper_err / a

    def pdf(self, x):
        """Функция плотности вероятности."""
        x = np.asarray(x, dtype=float) - self.median
        left = x <= 0
        right = ~left
        pdf = np.empty_like(x)
        pdf[left] = norm.pdf(x[left] / self.sigma_left) / self.sigma_left
        pdf[right] = norm.pdf(x[right] / self.sigma_right) / self.sigma_right
        return pdf

    def cdf(self, x):
        """Функция распределения."""
        x = np.asarray(x, dtype=float) - self.median
        left = x <= 0
        right = ~left
        cdf = np.empty_like(x)
        cdf[left] = norm.cdf(x[left] / self.sigma_left)
        cdf[right] = norm.cdf(x[right] / self.sigma_right)
        return cdf

    def ppf(self, q):
        """Квантильная функция (обратная к cdf)."""
        q = np.asarray(q)
        left = q <= 0.5
        right = ~left
        x = np.empty_like(q)
        x[left] = self.median + self.sigma_left * norm.ppf(q[left])
        x[right] = self.median + self.sigma_right * norm.ppf(q[right])
        return x

    def rvs(self, size=None, rng=None):
        """Генерация случайных выборок."""
        if rng is None:
            rng = np.random.default_rng()
        q = rng.uniform(size=size)
        return self.ppf(q)