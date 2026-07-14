"""
Реализации функций правдоподобия для совместного анализа сверхновых и GRB.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from abc import ABC, abstractmethod
import numpy as np
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core._numba_core import (mu_amati_mc_mode, mu_amati_with_errors,
                              loglike_grb_asym, chi2_grb_mc_mode, chi2_sn)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class BaseLikelihood(ABC):
    """Абстрактный базовый класс для вычислителя правдоподобия."""

    @abstractmethod
    def log_probability(self, theta, varying, fixed_vals, all_names, defaults):
        """Возвращает логарифмическую вероятность (или -inf)."""
        ...

class AmatiSNLikelihood(BaseLikelihood):
    """
    Совместное правдоподобие сверхновых (SN) и гамма-всплесков (GRB)
    с использованием корреляции Амати.

    Parameters
    ----------
    cosmo : Cosmology
        Космологическая модель.
    arrays : dict
        Словарь с подготовленными плоскими массивами (z_sn, mu_sn, ...).
    is_cloud : bool
        Используется ли облачный режим для GRB.
    """

    def __init__(self, cosmo, arrays, is_cloud):
        self.cosmo = cosmo
        self.arrays = arrays
        self.is_cloud = is_cloud

    def log_probability(self, theta, varying, fixed_vals, all_names, defaults):
        """
        Вычислить логарифм совместного правдоподобия SN + GRB.

        Parameters
        ----------
        theta : ndarray
            Текущие значения варьируемых параметров.
        varying : list of str
            Имена варьируемых параметров.
        fixed_vals : dict
            Словарь фиксированных параметров.
        all_names : list of str
            Полный список имён параметров (для совместимости).
        defaults : list
            Значения по умолчанию (не используются).

        Returns
        -------
        loglike : float
            Логарифмическая вероятность.
        """
        # Собираем полный словарь параметров
        p = dict(fixed_vals)
        for name, val in zip(varying, theta):
            p[name] = val

        # Обновляем космологию с использованием Ode
        H0 = p.get('H0', 70.0)
        Ode0 = p.get('Ode0', 0.7)
        w = p.get('w', -1.0)
        Ok0 = p.get('Ok0', 0.0)
        self.cosmo.update(H0=H0, Ode=Ode0, w=w, Ok=Ok0)

        loglike = 0.0
        arr = self.arrays

        # --- Сверхновые ---
        if len(arr['z_sn']) > 0:
            mu_model = self.cosmo.mu(arr['z_sn'])
            chi2 = chi2_sn(arr['mu_sn'], mu_model, arr['dmu_sn'])
            loglike += -0.5 * chi2

        # --- Гамма-всплески ---
        if len(arr['z_grb']) > 0:
            a = p.get('a', 1.0)
            b = p.get('b', 50.0)
            k = p.get('k', 0.0)
            mu_cosmo = self.cosmo.mu(arr['z_grb'])

            if self.is_cloud:
                mu_amati = mu_amati_mc_mode(
                    arr['z_grb'], arr['sbolo_grb'], arr['e_pi_grb'],
                    a, b, k
                )
                residuals = mu_cosmo - mu_amati
                chi2_raw = chi2_grb_mc_mode(residuals)
                n_orig = arr['n_grb_orig']
                n_total = len(arr['z_grb'])
                chi2 = chi2_raw * (n_orig / n_total)
                loglike += -0.5 * chi2
            else:
                sigma_int = p.get('sigma_int', 1.0)
                mu_amati, err_l, err_u = mu_amati_with_errors(
                    arr['z_grb'], arr['sbolo_grb'], arr['sbolo_err_grb'],
                    arr['e_pi_grb'], arr['e_pi_err_l_grb'], arr['e_pi_err_u_grb'],
                    a, b, k
                )
                residuals = mu_cosmo - mu_amati
                err_l_total = np.sqrt(err_l**2 + sigma_int**2)
                err_u_total = np.sqrt(err_u**2 + sigma_int**2)
                loglike += loglike_grb_asym(residuals, err_l_total, err_u_total)

        return loglike if np.isfinite(loglike) else -np.inf