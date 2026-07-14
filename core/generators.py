"""
Генераторы облаков точек для гамма-всплесков.
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
from core.distributions import SplitNormal
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class GRBCloudGenerator:
    """
    Генератор облаков точек для GRB на основе сплит-нормального распределения
    ошибок S_bolo и E_pi.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица с колонками:
        GRBname, z, sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u.
    """

    def __init__(self, df):
        self.df = df.copy()
        if (self.df[['sbolo', 'e_pi']] <= 0).any().any():
            raise ValueError("sbolo и e_pi должны быть > 0")
        if (self.df['sbolo'] - self.df['sbolo_err'] <= 0).any():
            raise ValueError("sbolo - sbolo_err должно быть > 0")
        if (self.df['e_pi'] - self.df['e_pi_err_l'] <= 0).any():
            raise ValueError("e_pi - e_pi_err_l должно быть > 0")

        self._grb_names = df['GRBname'].values
        self._z = df['z'].values
        self._sbolo = df['sbolo'].values
        self._sbolo_err = df['sbolo_err'].values
        self._e_pi = df['e_pi'].values
        self._e_pi_err_l = df['e_pi_err_l'].values
        self._e_pi_err_u = df['e_pi_err_u'].values

        self._n_points = None
        self._clouds = None
        self._compute_log_params()

    def _compute_log_params(self):
        """Вычислить параметры в логарифмическом пространстве."""
        sbolo = self._sbolo
        sbolo_err = self._sbolo_err
        self._log_sbolo_med = np.log10(sbolo)
        log_sbolo_low = np.log10(sbolo - sbolo_err)
        log_sbolo_high = np.log10(sbolo + sbolo_err)
        self._log_sbolo_err_left = self._log_sbolo_med - log_sbolo_low
        self._log_sbolo_err_right = log_sbolo_high - self._log_sbolo_med

        epi = self._e_pi
        epi_err_l = self._e_pi_err_l
        epi_err_u = self._e_pi_err_u
        self._log_epi_med = np.log10(epi)
        log_epi_low = np.log10(epi - epi_err_l)
        log_epi_high = np.log10(epi + epi_err_u)
        self._log_epi_err_left = self._log_epi_med - log_epi_low
        self._log_epi_err_right = log_epi_high - self._log_epi_med

    def generate_clouds(self, n_points=1000, seed=None, force=False):
        """
        Сгенерировать (или вернуть уже существующие) облака точек.

        Parameters
        ----------
        n_points : int
            Количество точек на один GRB.
        seed : int, optional
            Seed для генератора случайных чисел.
        force : bool
            Если True, перегенерировать облака даже при их наличии.

        Returns
        -------
        clouds : list of dict
            Список словарей с полями:
            GRBname, z, sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u,
            sbolo_mc (ndarray), e_pi_mc (ndarray).
        """
        if self._clouds is not None and not force:
            return self._clouds

        rng = np.random.default_rng(seed)
        clouds = []

        for i in range(len(self._grb_names)):
            dist_log_s = SplitNormal(
                median=self._log_sbolo_med[i],
                lower_err=self._log_sbolo_err_left[i],
                upper_err=self._log_sbolo_err_right[i]
            )
            dist_log_e = SplitNormal(
                median=self._log_epi_med[i],
                lower_err=self._log_epi_err_left[i],
                upper_err=self._log_epi_err_right[i]
            )

            log_s_mc = dist_log_s.rvs(size=n_points, rng=rng)
            log_e_mc = dist_log_e.rvs(size=n_points, rng=rng)

            entry = {
                'GRBname': self._grb_names[i],
                'z': self._z[i],
                'sbolo': self._sbolo[i],
                'sbolo_err': self._sbolo_err[i],
                'e_pi': self._e_pi[i],
                'e_pi_err_l': self._e_pi_err_l[i],
                'e_pi_err_u': self._e_pi_err_u[i],
                'sbolo_mc': np.power(10, log_s_mc),
                'e_pi_mc': np.power(10, log_e_mc)
            }
            clouds.append(entry)

        self._clouds = clouds
        self._n_points = n_points
        return clouds
