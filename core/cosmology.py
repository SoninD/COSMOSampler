"""
Космологическая модель Фридмана, параметризованная плотностью тёмной энергии.

Основной класс Cosmology хранит H0, Ode, Ok, w, а плотность материи Om
автоматически вычисляется из условия Om + Ok + Ode = 1.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core._numba_core import (hubble_norm_array, integral_array,
                              d_M_array, d_L_array, d_E_array, distance_modulus_array)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Cosmology:
    """Космологическая модель Фридмана.

    Параметры
    ----------
    H0 : float
        Постоянная Хаббла (км/с/Мпк). По умолчанию 70.
    Ode : float
        Параметр плотности тёмной энергии. По умолчанию 0.7.
    w : float
        Уравнение состояния тёмной энергии. По умолчанию -1.
    Ok : float
        Параметр кривизны. По умолчанию 0.
    """

    def __init__(self, H0=70.0, Ode=0.7, w=-1.0, Ok=0.0):
        self.H0 = float(H0)
        self.Ode = float(Ode)
        self.w = float(w)
        self.Ok = float(Ok)
        self.Om = 1.0 - self.Ok - self.Ode

    def update(self, H0=None, Ode=None, Ok=None, w=None):
        """Обновить космологические параметры."""
        if H0 is not None:
            self.H0 = H0
        if Ok is not None:
            self.Ok = Ok
        if w is not None:
            self.w = w
        if Ode is not None:
            self.Ode = Ode
        # В любом случае пересчитываем Om
        self.Om = 1.0 - self.Ok - self.Ode

    # ---- Базовые функции ----
    def E(self, z, n_points=300):
        """Нормированный параметр Хаббла h(z) = H(z)/H0."""
        scalar = np.isscalar(z)
        z_arr = np.array([z]) if scalar else np.asarray(z)
        res = hubble_norm_array(z_arr, self.Ode, self.Ok, self.w)   # исправлено
        return res[0] if scalar else res

    def I(self, z, n_points=300):
        """Интеграл I(z) = ∫₀ᶻ dz' / h(z')."""
        scalar = np.isscalar(z)
        z_arr = np.array([z]) if scalar else np.asarray(z)
        res = integral_array(z_arr, self.Ode, self.Ok, self.w, n_points)  # исправлено
        return res[0] if scalar else res

    # ---- Расстояния ----
    def d_M(self, z, n_points=300):
        """Поперечное сопутствующее расстояние (Мпк)."""
        scalar = np.isscalar(z)
        z_arr = np.array([z]) if scalar else np.asarray(z)
        res = d_M_array(z_arr, self.H0, self.Ode, self.Ok, self.w, n_points)  # исправлено
        return res[0] if scalar else res

    def d_L(self, z, n_points=300):
        """Яркостное расстояние (Мпк)."""
        scalar = np.isscalar(z)
        z_arr = np.array([z]) if scalar else np.asarray(z)
        res = d_L_array(z_arr, self.H0, self.Ode, self.Ok, self.w, n_points)  # исправлено
        return res[0] if scalar else res

    def d_E(self, z, n_points=300):
        """Энергетическое расстояние (Мпк)."""
        scalar = np.isscalar(z)
        z_arr = np.array([z]) if scalar else np.asarray(z)
        res = d_E_array(z_arr, self.H0, self.Ode, self.Ok, self.w, n_points)  # исправлено
        return res[0] if scalar else res

    def distance_modulus(self, z, n_points=300):
        """Модуль расстояния μ(z) = 5 log₁₀(d_L / 10 пк)."""
        scalar = np.isscalar(z)
        z_arr = np.array([z]) if scalar else np.asarray(z)
        res = distance_modulus_array(z_arr, self.H0, self.Ode, self.Ok, self.w, n_points)  # исправлено
        return res[0] if scalar else res

    def mu(self, z, n_points=300):
        """Алиас для distance_modulus."""
        return self.distance_modulus(z, n_points)

    def __repr__(self):
        return (f"Cosmology(H0={self.H0:.3f}, Ode={self.Ode:.3f}, "
                f"Ok={self.Ok:.3f}, w={self.w:.3f})")

