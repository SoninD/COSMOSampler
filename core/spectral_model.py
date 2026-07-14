"""
Абстрактная спектральная модель и её реализации для вычисления
S_bolo и E_pi в системе покоя гамма-всплеска.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
logger = logging.getLogger(__name__)


class BaseSpectralModel(ABC):
    """
    Абстрактная спектральная модель.

    Предоставляет общий интерфейс для вычисления болометрической плотности
    потока S_bolo и пиковой энергии E_pi в системе покоя для заданного
    гамма-всплеска.

    Parameters
    ----------
    catalogue : pandas.DataFrame
        Каталог GRB, содержащий спектральные параметры и наблюдаемые потоки.
    """

    def __init__(self, catalogue: pd.DataFrame):
        self.cat = catalogue

    @abstractmethod
    def spectrum(self, energy, params):
        """
        Вычислить поток N(E) (в произвольных единицах) для массива энергий.

        Parameters
        ----------
        energy : ndarray
            Массив значений энергии.
        params : list
            Список параметров модели, извлечённый из строки каталога.

        Returns
        -------
        flux : ndarray
            Поток N(E) для каждого значения энергии.
        """
        ...

    def get_params(self, row):
        """
        Извлечь параметры модели из строки каталога.

        Parameters
        ----------
        row : pandas.Series
            Строка каталога с необходимыми колонками.

        Returns
        -------
        params : list
            Список параметров модели.
        """
        return []

    def energy_int(self, int_lims, params):
        """
        Вычислить определённый интеграл от spectrum в заданных пределах.

        Parameters
        ----------
        int_lims : tuple (float, float)
            Нижний и верхний пределы интегрирования.
        params : list
            Параметры модели.

        Returns
        -------
        integral : float
            Значение интеграла.
        """
        e_range = np.linspace(int_lims[0], int_lims[1], num=5000)
        en_range = self.spectrum(e_range, params)
        return np.trapezoid(en_range, e_range)

    def s_bolo(self, z, params, s_obs, s_obs_err):
        """
        Вычислить болометрическую плотность потока S_bolo.

        Parameters
        ----------
        z : float
            Красное смещение.
        params : list
            Параметры модели.
        s_obs : float
            Наблюдаемая плотность потока.
        s_obs_err : float
            Ошибка наблюдаемой плотности потока.

        Returns
        -------
        S_bolo : float
            Болометрическая плотность потока.
        S_bolo_err : float
            Ошибка болометрической плотности потока.
        """
        num_int = self.energy_int((1 / (1 + z), 1e4 / (1 + z)), params)
        denom_int = self.energy_int((1, 150), params)
        if denom_int == 0:
            return np.inf, 0
        return s_obs * num_int / denom_int, s_obs_err * num_int / denom_int

    def energy_peak_rest_frame(self, z, energy_peak, energy_peak_err_l, energy_peak_err_u):
        """
        Пересчитать пиковую энергию в систему покоя.

        Parameters
        ----------
        z : float
            Красное смещение.
        energy_peak : float
            Пиковая энергия в системе наблюдателя.
        energy_peak_err_l : float
            Нижняя ошибка пиковой энергии.
        energy_peak_err_u : float
            Верхняя ошибка пиковой энергии.

        Returns
        -------
        e_pi : float
            Пиковая энергия в системе покоя.
        e_pi_err_l : float
            Нижняя ошибка пиковой энергии в системе покоя.
        e_pi_err_u : float
            Верхняя ошибка пиковой энергии в системе покоя.
        """
        return (energy_peak * (1 + z),
                energy_peak_err_l * (1 + z),
                energy_peak_err_u * (1 + z))

    def get_s_e_data(self, grb_name=None):
        """
        Построить таблицу с S_bolo и E_pi для выбранных GRB.

        Parameters
        ----------
        grb_name : str or list of str, optional
            Имя или список имён GRB. Если None, обрабатываются все.

        Returns
        -------
        result : pandas.DataFrame
            Таблица с колонками:
            GRBname, z, sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u, T90.
        """
        if grb_name is None:
            data_to_process = self.cat.copy()
        elif isinstance(grb_name, str):
            data_to_process = self.cat[self.cat['GRBname'] == grb_name].copy()
            if data_to_process.empty:
                return pd.DataFrame()
        elif isinstance(grb_name, list):
            data_to_process = self.cat[self.cat['GRBname'].isin(grb_name)].copy()
            if data_to_process.empty:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        sbolo_list, sbolo_err_list, e_pi_list = [], [], []
        e_pi_err_l_list, e_pi_err_u_list = [], []
        name_list, z_list, t90_list = [], [], []

        for idx, row in data_to_process.iterrows():
            try:
                S_obs, S_obs_err = row['S_obs'], row['S_obs_err']
                redshift = row['redshift']
                DEpeak_hi, DEpeak_low = row.get('DEpeak_hi', 0), row.get('DEpeak_low', 0)
                Epeak = row.get('Epeak', 1)
                name = row['GRBname']
                t90 = row.get('T90', np.nan)

                params = self.get_params(row)

                S_bolo, S_bolo_err = self.s_bolo(redshift, params, S_obs, S_obs_err)
                e_pi, e_pi_err_l, e_pi_err_u = self.energy_peak_rest_frame(
                    redshift, Epeak, DEpeak_low, DEpeak_hi
                )
                sbolo_list.append(S_bolo)
                sbolo_err_list.append(S_bolo_err)
                e_pi_list.append(e_pi)
                e_pi_err_l_list.append(e_pi_err_l)
                e_pi_err_u_list.append(e_pi_err_u)
                name_list.append(name)
                z_list.append(redshift)
                t90_list.append(t90)
            except Exception as e:
                logger.error(f"Ошибка при обработке GRB {name}: {e}")
                continue

        if name_list:
            return pd.DataFrame({
                'GRBname': name_list, 'z': z_list,
                'sbolo': sbolo_list, 'sbolo_err': sbolo_err_list,
                'e_pi': e_pi_list, 'e_pi_err_l': e_pi_err_l_list,
                'e_pi_err_u': e_pi_err_u_list, 'T90': t90_list
            })
        else:
            return pd.DataFrame()


class CPLModel(BaseSpectralModel):
    """
    Модель Cutoff Power Law (CPL) для спектра гамма-всплеска.

    N(E) = A_norm * (E / E_norm)^alpha * exp(-E * (2 + alpha) / E_peak).

    Parameters
    ----------
    catalogue : pandas.DataFrame
        Каталог GRB, содержащий колонки:
        alpha, Epeak, enorm, norm, S_obs, S_obs_err, redshift, DEpeak_hi, DEpeak_low.
    """

    def spectrum(self, energy, params):
        """
        Вычислить поток CPL для массива энергий.

        Parameters
        ----------
        energy : ndarray
            Массив энергий.
        params : list [alpha, E_peak, E_norm, A_norm]
            Параметры модели.

        Returns
        -------
        flux : ndarray
            N(E).
        """
        alpha, energy_peak, energy_norm, norm_const = params
        return (norm_const * (energy / energy_norm) ** alpha *
                np.exp(-energy * (2 + alpha) / energy_peak))

    def get_params(self, row):
        """
        Извлечь параметры CPL из строки каталога.

        Parameters
        ----------
        row : pandas.Series
            Строка с колонками 'alpha', 'Epeak', 'enorm', 'norm'.

        Returns
        -------
        params : list
            [alpha, E_peak, E_norm, A_norm].
        """
        return [row['alpha'], row['Epeak'], row['enorm'], row['norm']]
