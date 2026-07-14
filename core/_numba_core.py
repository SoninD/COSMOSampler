"""
Низкоуровневые ускоренные функции (Numba) для космологических вычислений
и статистик гамма-всплесков / сверхновых.
Все функции оптимизированы с помощью декоратора @njit.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
from numba import njit, prange
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core import constants as const
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# --------------------------------------------------------------
# 1. Нормированный параметр Хаббла h(z) для массива
# --------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def hubble_norm_array(z, Ode, Ok, w):
    """
    Нормированный параметр Хаббла h(z) = H(z)/H0 для массива красных смещений.

    Parameters
    ----------
    z : ndarray
        Массив красных смещений.
    Ode : float
        Параметр плотности тёмной энергии.
    Ok : float
        Параметр кривизны.
    w : float
        Уравнение состояния тёмной энергии.

    Returns
    -------
    out : ndarray
        Значения h(z) для каждого элемента z.
    """
    n = len(z)
    out = np.empty(n, dtype=np.float64)
    Om = 1.0 - Ok - Ode               # плотность материи из условия Фридмана
    for i in prange(n):
        zi = z[i]
        out[i] = np.sqrt(Om*(1+zi)**3 + Ok*(1+zi)**2 + Ode*(1+zi)**(3*(1+w)))
    return out

# --------------------------------------------------------------
# 2. Интеграл I(z) = ∫₀ᶻ dz' / h(z') для массива
# --------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def integral_array(z, Ode, Ok, w, n_points=300):
    """
    Безразмерный интеграл расстояния I(z) для массива z.

    Parameters
    ----------
    z : ndarray
        Красные смещения.
    Ode : float
        Параметр плотности тёмной энергии.
    Ok : float
        Параметр кривизны.
    w : float
        Уравнение состояния тёмной энергии.
    n_points : int, optional
        Число точек для численного интегрирования.

    Returns
    -------
    out : ndarray
        Значения I(z) для каждого z.
    """
    n = len(z)
    out = np.empty(n, dtype=np.float64)
    Om = 1.0 - Ok - Ode
    for i in prange(n):
        zi = z[i]
        if zi <= 0.0:
            out[i] = 0.0
            continue
        dz = zi / n_points
        integral = 0.0
        for j in range(n_points):
            zj = j * dz
            if zj >= zi:
                break
            h = np.sqrt(Om*(1+zj)**3 + Ok*(1+zj)**2 + Ode*(1+zj)**(3*(1+w)))
            integral += dz / h
        out[i] = integral
    return out

# --------------------------------------------------------------
# 3. Поперечное сопутствующее расстояние d_M(z) для массива
# --------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def d_M_array(z, H0, Ode, Ok, w, n_points=300):
    """
    Поперечное сопутствующее расстояние (Мпк) для массива z.

    Parameters
    ----------
    z : ndarray
        Красные смещения.
    H0 : float
        Постоянная Хаббла (км/с/Мпк).
    Ode : float
        Параметр плотности тёмной энергии.
    Ok : float
        Параметр кривизны.
    w : float
        Уравнение состояния тёмной энергии.
    n_points : int, optional
        Число точек для интегрирования.

    Returns
    -------
    out : ndarray
        Поперечное сопутствующее расстояние в Мпк.
    """
    n = len(z)
    out = np.empty(n, dtype=np.float64)
    Om = 1.0 - Ok - Ode
    c_over_H0 = const.C_KM_S / H0
    for i in prange(n):
        zi = z[i]
        if zi <= 0.0:
            out[i] = 0.0
            continue
        dz = zi / n_points
        integral = 0.0
        for j in range(n_points):
            zj = j * dz
            if zj >= zi:
                break
            h = np.sqrt(Om*(1+zj)**3 + Ok*(1+zj)**2 + Ode*(1+zj)**(3*(1+w)))
            integral += dz / h
        if abs(Ok) < 1e-10:
            out[i] = c_over_H0 * integral
        elif Ok > 0:
            sqrtOk = np.sqrt(Ok)
            out[i] = c_over_H0 * np.sinh(sqrtOk * integral) / sqrtOk
        else:
            sqrtNegOk = np.sqrt(-Ok)
            out[i] = c_over_H0 * np.sin(sqrtNegOk * integral) / sqrtNegOk
    return out

# --------------------------------------------------------------
# 4. Яркостное расстояние d_L(z) для массива
# --------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def d_L_array(z, H0, Ode, Ok, w, n_points=300):
    """
    Яркостное расстояние (Мпк) для массива z.

    Parameters
    ----------
    z : ndarray
        Красные смещения.
    H0 : float
        Постоянная Хаббла.
    Ode : float
        Параметр плотности тёмной энергии.
    Ok : float
        Параметр кривизны.
    w : float
        Уравнение состояния тёмной энергии.
    n_points : int, optional
        Число точек для интегрирования.

    Returns
    -------
    out : ndarray
        Яркостное расстояние в Мпк.
    """
    d_M = d_M_array(z, H0, Ode, Ok, w, n_points)
    return (1.0 + z) * d_M

# --------------------------------------------------------------
# 5. Энергетическое расстояние d_E(z) для массива
# --------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def d_E_array(z, H0, Ode, Ok, w, n_points=300):
    """
    Энергетическое расстояние (Мпк) для массива z.

    Parameters
    ----------
    z : ndarray
        Красные смещения.
    H0 : float
        Постоянная Хаббла.
    Ode : float
        Параметр плотности тёмной энергии.
    Ok : float
        Параметр кривизны.
    w : float
        Уравнение состояния тёмной энергии.
    n_points : int, optional
        Число точек для интегрирования.

    Returns
    -------
    out : ndarray
        Энергетическое расстояние в Мпк.
    """
    d_M = d_M_array(z, H0, Ode, Ok, w, n_points)
    return np.sqrt(1.0 + z) * d_M

# --------------------------------------------------------------
# 6. Модуль расстояния μ(z) для массива
# --------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def distance_modulus_array(z, H0, Ode, Ok, w, n_points=300):
    """
    Модуль расстояния μ(z) = 5 log₁₀(d_L / 10 пк) для массива z.

    Parameters
    ----------
    z : ndarray
        Красные смещения.
    H0 : float
        Постоянная Хаббла.
    Ode : float
        Параметр плотности тёмной энергии.
    Ok : float
        Параметр кривизны.
    w : float
        Уравнение состояния тёмной энергии.
    n_points : int, optional
        Число точек для интегрирования.

    Returns
    -------
    mu : ndarray
        Модуль расстояния.
    """
    dL = d_L_array(z, H0, Ode, Ok, w, n_points)
    mu = 25.0 + 5.0 * np.log10(dL)
    return mu

# --------------------------------------------------------------
# 7. Функции для сверхновых и гамма-всплесков
# --------------------------------------------------------------

@njit(fastmath=True, cache=True)
def mu_amati_mc_mode(z, sbolo, e_pi, a, b, k):
    """
    Модуль расстояния по корреляции Амати (облачный режим).

    Parameters
    ----------
    z : ndarray
        Красные смещения.
    sbolo : ndarray
        Болометрическая плотность потока [эрг/см²/с].
    e_pi : ndarray
        Пиковая энергия в системе покоя [кэВ].
    a, b, k : float
        Параметры корреляции Амати.

    Returns
    -------
    mu : ndarray
        Модуль расстояния.
    """
    n = z.size
    mu = np.empty(n, dtype=np.float64)
    for i in range(n):
        term1 = np.log10((z[i]+1)**(k+1) / (4*np.pi * sbolo[i]) / const.MPC_IN_CM**2)
        mu[i] = 25.0 + 2.5 * (term1 + a * np.log10(e_pi[i]) + b)
    return mu

@njit(fastmath=True, cache=True)
def mu_amati_with_errors(z, sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u,
                         a, b, k):
    """
    Модуль расстояния и асимметричные ошибки по корреляции Амати.

    Parameters
    ----------
    z : ndarray
        Красные смещения.
    sbolo : ndarray
        Болометрическая плотность потока [эрг/см²/с].
    sbolo_err : ndarray
        Ошибка болометрической плотности потока.
    e_pi : ndarray
        Пиковая энергия в системе покоя [кэВ].
    e_pi_err_l, e_pi_err_u : ndarray
        Нижняя и верхняя ошибки пиковой энергии.
    a, b, k : float
        Параметры корреляции Амати.

    Returns
    -------
    mu : ndarray
        Модуль расстояния.
    err_l : ndarray
        Нижняя ошибка.
    err_u : ndarray
        Верхняя ошибка.
    """
    n = z.size
    mu = np.empty(n, dtype=np.float64)
    err_l = np.empty(n, dtype=np.float64)
    err_u = np.empty(n, dtype=np.float64)
    log10_const = 2.5 / np.log(10.0)
    for i in range(n):
        term1 = np.log10((z[i]+1)**(k+1) / (4*np.pi * sbolo[i]) / const.MPC_IN_CM**2)
        mu[i] = 25.0 + 2.5 * (term1 + a * np.log10(e_pi[i]) + b)
        dmuds = -log10_const / sbolo[i]
        abs_a = abs(a)
        dmude_abs = log10_const * abs_a / e_pi[i]
        err_l[i] = np.sqrt((dmuds * sbolo_err[i])**2 + (dmude_abs * e_pi_err_l[i])**2)
        err_u[i] = np.sqrt((dmuds * sbolo_err[i])**2 + (dmude_abs * e_pi_err_u[i])**2)
    return mu, err_l, err_u

@njit(fastmath=True, cache=True)
def chi2_sn(mu_obs, mu_model, dmu):
    """
    χ² для сверхновых с симметричными ошибками.

    Parameters
    ----------
    mu_obs : ndarray
        Наблюдаемые модули расстояния.
    mu_model : ndarray
        Модельные модули расстояния.
    dmu : ndarray
        Ошибки измерений.

    Returns
    -------
    chi2 : float
        Значение χ².
    """
    diff = mu_obs - mu_model
    chi2 = 0.0
    for i in range(diff.size):
        chi2 += (diff[i] / dmu[i]) ** 2
    return chi2


@njit(fastmath=True, cache=True)
def chi2_grb_mc_mode(residuals):
    """
    χ² для GRB в облачном режиме (веса одинаковы).

    Parameters
    ----------
    residuals : ndarray
        Разности (данные – модель).

    Returns
    -------
    chi2 : float
        Значение χ².
    """
    chi2 = 0.0
    for i in range(residuals.size):
        chi2 += residuals[i] * residuals[i]
    return chi2

@njit(fastmath=True, cache=True)
def loglike_grb_asym(residuals, err_l, err_u):
    """
    Логарифм правдоподобия для GRB с асимметричными ошибками.

    Parameters
    ----------
    residuals : ndarray
        Разности (данные – модель).
    err_l : ndarray
        Нижние ошибки.
    err_u : ndarray
        Верхние ошибки.

    Returns
    -------
    logL : float
        Значение логарифма правдоподобия.
    """
    chi2 = 0.0
    norm = 0.0
    for i in range(residuals.size):
        sigma = err_u[i] if residuals[i] >= 0 else err_l[i]
        chi2 += (residuals[i] / sigma) ** 2
        norm += np.log(err_l[i] + err_u[i])
    return -0.5 * chi2 - norm