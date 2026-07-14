"""
Модуль визуализации результатов MCMC-анализа (класс ResultPlotter).

Строит диаграмму Хаббла, trace plots, posterior distributions,
corner plot и сохраняет их в папку результатов.
Все графики используют единый словарь научных обозначений параметров.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
import corner
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.distributions import SplitNormal
from core._numba_core import (
    mu_amati_mc_mode,
    mu_amati_with_errors,
    chi2_grb_mc_mode,
)
from core import constants as const
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Единый словарь научных обозначений параметров
# ---------------------------------------------------------------------------
PARAM_LABELS = {
    'H0': r'$H_0$',
    'Ode0': r'$\Omega_{de}$',
    'w': r'$w$',
    'Ok0': r'$\Omega_k$',
    'a': r'$a$',
    'b': r'$b$',
    'k': r'$k$',
    'sigma_int': r'$\sigma_{\rm int}$'
}

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def _free_param_count(param_names, fixed):
    """Количество свободных параметров из списка param_names."""
    return len([p for p in param_names if p not in fixed])


def _compute_dof(n_data, free_params):
    """Число степеней свободы (минимум 1)."""
    return max(n_data - free_params, 1)


def _percentiles(vals):
    """Возвращает (16%, 50%, 84%) для массива семплов."""
    return np.percentile(vals, [16, 50, 84])


def _label_with_errors(param_name, p16, p50, p84, fmt='.2f'):
    """Строит строку вида '$Label = value^{+up}_{-low}$'."""
    label = PARAM_LABELS.get(param_name, param_name)
    up = p84 - p50
    low = p50 - p16
    return f'{label} = ${p50:{fmt}}^{{+{up:{fmt}}}}_{{-{low:{fmt}}}}$'


# ---------------------------------------------------------------------------
# ResultPlotter
# ---------------------------------------------------------------------------
class ResultPlotter:
    """Построение всех основных графиков на основе результатов FitterModel.

    Публичные методы (plot_hubble_diagram, plot_trace, plot_posterior_distributions,
    plot_corner) могут принимать внешние оси (axes) для встраивания в GUI.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : FitterModel
            Обученная модель, содержащая samples, chain, median_params и т.д.
        """
        self.model = model
        self.results_path = model.results_path

    # -----------------------------------------------------------------
    # Внутренние утилиты для Hubble diagram
    # -----------------------------------------------------------------
    def _setup_cosmology(self, mp):
        """Обновить космологию из медианных параметров и вернуть объект."""
        H0 = mp.get('H0', 70.0)
        Ode0 = mp.get('Ode0', 0.7)
        w = mp.get('w', -1.0)
        Ok0 = mp.get('Ok0', 0.0)
        cosmo = self.model.cosmo
        cosmo.update(H0=H0, Ode=Ode0, w=w, Ok=Ok0)
        return cosmo

    def _get_z_range(self, arr):
        """Вернуть объединённый массив z (SN + GRB) и гладкую сетку."""
        z_parts = []
        if len(arr['z_sn']) > 0:
            z_parts.append(arr['z_sn'])
        if len(arr['z_grb']) > 0:
            z_parts.append(arr['z_grb'])
        if not z_parts:
            return None, None
        z_all = np.concatenate(z_parts)
        z_smooth = np.linspace(z_all.min() * 0.9, z_all.max() * 1.1, 200)
        return z_all, z_smooth

    def _compute_dof_and_labels(self, arr, mp):
        """Вычислить степени свободы и текстовые метки для легенды."""
        model = self.model
        data = {'dof_sn': 0, 'dof_grb': 0, 'parts': []}
        free_cosm = _free_param_count(['H0', 'Ode0', 'w', 'Ok0'], model.fixed)

        # SN
        z_sn = arr['z_sn']
        if len(z_sn) > 0:
            n_sn = len(z_sn)
            data['dof_sn'] = _compute_dof(n_sn, free_cosm)
            data['parts'].append(f'SN: {n_sn} (dof={data["dof_sn"]})')

        # GRB
        if arr.get('is_cloud', False):
            n_cloud = len(arr['z_grb'])
            free_amati = _free_param_count(['a', 'b', 'k'], model.fixed)
            free_cosm_grb = free_cosm
            dof_cloud = _compute_dof(n_cloud, free_amati + free_cosm_grb)
            dof_grb_orig = _compute_dof(len(model.grb_df), free_amati + free_cosm_grb)
            data['dof_grb'] = dof_cloud
            data['parts'].append(f'GRB clouds: {n_cloud} (dof={dof_cloud})')
            data['parts'].append(f'GRB orig: {len(model.grb_df)}')
            data['dof_grb_orig'] = dof_grb_orig
        elif len(arr['z_grb']) > 0:
            n_grb = len(arr['z_grb'])
            free_amati = _free_param_count(['a', 'b', 'k'], model.fixed)
            if 'sigma_int' in mp and 'sigma_int' not in model.fixed:
                free_amati += 1
            free_cosm_grb = free_cosm
            dof_grb = _compute_dof(n_grb, free_amati + free_cosm_grb)
            data['dof_grb'] = dof_grb
            data['parts'].append(f'GRB: {n_grb} (dof={dof_grb})')

        return data

    def _plot_theoretical_curves(self, ax, z_smooth, cosmo):
        """Нарисовать теоретические кривые ΛCDM и wCDM."""
        # wCDM (median)
        mu_wcdm = cosmo.mu(z_smooth)
        ax.plot(z_smooth, mu_wcdm, 'k--', lw=2, label='wCDM (median)')
        # ΛCDM (flat, w=-1)
        cosmo_flat = copy.deepcopy(self.model.cosmo)
        cosmo_flat.update(H0=70, Ode=0.7, Ok=0.0, w=-1.0)   # плоская ΛCDM
        mu_lcdm = cosmo_flat.mu(z_smooth)
        ax.plot(z_smooth, mu_lcdm, 'g-', lw=1.5, alpha=0.7, label=r'$\Lambda$CDM (flat, w=-1)')
        # Вернуть медианные параметры
        mp = self.model.median_params
        cosmo.update(H0=mp.get('H0', 70.0),
                     Ode=mp.get('Ode0', 0.7),
                     w=mp.get('w', -1.0),
                     Ok=mp.get('Ok0', 0.0))

    def _plot_sn_on_hubble(self, ax1, ax2, arr, cosmo):
        """Нарисовать сверхновые на диаграмме Хаббла и остатках."""
        z_sn = arr['z_sn']
        if len(z_sn) == 0:
            return None
        mu_sn = arr['mu_sn']
        dmu_sn = arr['dmu_sn']
        mu_model = cosmo.mu(z_sn)
        ax1.errorbar(z_sn, mu_sn, yerr=dmu_sn, fmt='o', markersize=1, alpha=0.8,
                     color='purple', markeredgecolor='blue', markerfacecolor='blue',
                     label='Pantheon SN', capsize=4, elinewidth=1)
        res = mu_sn - mu_model
        chi2 = np.sum((res / dmu_sn) ** 2)
        dof_sn = _compute_dof(len(z_sn),
                              _free_param_count(['H0', 'Ode0', 'w', 'Ok0'], self.model.fixed))
        ax2.errorbar(z_sn, res, yerr=dmu_sn, fmt='s', markersize=1, alpha=0.8,
                     color='purple', markeredgecolor='blue', capsize=2, elinewidth=1,
                     label=rf'SN: $\chi^2/dof = {chi2:.1f}/{dof_sn}$')
        return chi2

    def _plot_grb_clouds_on_hubble(self, ax1, ax2, arr, mp, cosmo):
        """Нарисовать облака точек GRB и исходные данные, вернуть χ²."""
        a = mp.get('a', 1.0)
        b = mp.get('b', 50.0)
        k = mp.get('k', 0.0)
        model = self.model

        # Облака
        mu_cloud = mu_amati_mc_mode(arr['z_grb'], arr['sbolo_grb'], arr['e_pi_grb'], a, b, k)
        ax1.scatter(arr['z_grb'], mu_cloud, s=1, alpha=0.1, color='gray', label='MC clouds')
        mu_cosmo_cloud = cosmo.mu(arr['z_grb'])
        res_cloud = mu_cloud - mu_cosmo_cloud
        ax2.scatter(arr['z_grb'], res_cloud, s=1, alpha=0.1, color='gray')
        chi2_cloud = chi2_grb_mc_mode(res_cloud) * (arr['n_grb_orig'] / len(arr['z_grb']))

        # Исходные GRB
        grb_orig = model.grb_df
        z_orig = grb_orig['z'].values.astype(np.float64)
        sbolo = grb_orig['sbolo'].values.astype(np.float64)
        sbolo_err = grb_orig['sbolo_err'].values.astype(np.float64)
        e_pi = grb_orig['e_pi'].values.astype(np.float64)
        e_pi_err_l = grb_orig['e_pi_err_l'].values.astype(np.float64)
        e_pi_err_u = grb_orig['e_pi_err_u'].values.astype(np.float64)

        mu_orig, err_l, err_u = mu_amati_with_errors(
            z_orig, sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u, a, b, k
        )
        yerr = np.array([err_l, err_u])
        ax1.errorbar(z_orig, mu_orig, yerr=yerr, fmt='o', markersize=2, alpha=0.8,
                     color='orange', markeredgecolor='red', markerfacecolor='red',
                     label=f'GRB orig (a={a:.2f}, b={b:.2f}, k={k:.2f})', capsize=4, elinewidth=1)
        mu_model_orig = cosmo.mu(z_orig)
        res_orig = mu_orig - mu_model_orig
        sigma = np.where(res_orig >= 0, err_u, err_l)
        chi2_orig = np.sum((res_orig / sigma) ** 2)
        dof_orig = _compute_dof(len(z_orig),
                                _free_param_count(['a', 'b', 'k'], model.fixed) +
                                _free_param_count(['H0', 'Ode0', 'w', 'Ok0'], model.fixed))
        ax2.errorbar(z_orig, res_orig, yerr=yerr, fmt='o', markersize=2, alpha=0.8,
                     color='orange', markeredgecolor='red', capsize=2, elinewidth=1,
                     label=rf'GRB orig: $\chi^2/dof = {chi2_orig:.1f}/{dof_orig}$')
        return chi2_cloud

    def _plot_grb_sigma_int_on_hubble(self, ax1, ax2, arr, mp, cosmo):
        """Нарисовать GRB в режиме sigma_int и вернуть χ²."""
        a = mp.get('a', 1.0)
        b = mp.get('b', 50.0)
        k = mp.get('k', 0.0)
        sigma_int = mp.get('sigma_int', 0.0)
        model = self.model

        mu_grb, err_l_raw, err_u_raw = mu_amati_with_errors(
            arr['z_grb'], arr['sbolo_grb'], arr['sbolo_err_grb'],
            arr['e_pi_grb'], arr['e_pi_err_l_grb'], arr['e_pi_err_u_grb'],
            a, b, k
        )
        err_l = np.sqrt(err_l_raw**2 + sigma_int**2)
        err_u = np.sqrt(err_u_raw**2 + sigma_int**2)
        yerr = np.array([err_l, err_u])

        ax1.errorbar(arr['z_grb'], mu_grb, yerr=yerr, fmt='o', markersize=2, alpha=0.8,
                     color='orange', markeredgecolor='red', markerfacecolor='red',
                     label = 'GRB (a={:.2f}, b={:.1f}, k={:.2f}, '.format(a, b, k) + r'$\sigma_{\rm int}$' + ' = {:.2f})'.format(sigma_int), capsize=4, elinewidth=1)
        mu_model = cosmo.mu(arr['z_grb'])
        res = mu_grb - mu_model
        sigma = np.where(res >= 0, err_u, err_l)
        chi2 = np.sum((res / sigma)**2)
        dof_grb = _compute_dof(len(arr['z_grb']),
                               _free_param_count(['a', 'b', 'k'], model.fixed) +
                               (1 if 'sigma_int' not in model.fixed else 0) +
                               _free_param_count(['H0', 'Ode0', 'w', 'Ok0'], model.fixed))
        ax2.errorbar(arr['z_grb'], res, yerr=yerr, fmt='o', markersize=2, alpha=0.8,
                     color='orange', markeredgecolor='red', capsize=2, elinewidth=1,
                     label=rf'GRB: $\chi^2/dof = {chi2:.1f}/{dof_grb}$')
        return chi2

    # -----------------------------------------------------------------
    # Публичные методы визуализации
    # -----------------------------------------------------------------
    def plot_hubble_diagram(self, figsize=(12, 10), save=True, axes=None):
        """Построить диаграмму Хаббла (μ(z)) и остатки относительно ΛCDM.

        Parameters
        ----------
        figsize : tuple
            Размер фигуры (если axes не заданы).
        save : bool
            Сохранить ли график в файл.
        axes : tuple of matplotlib.axes.Axes, optional
            Кортеж (ax_main, ax_residuals) для отрисовки на существующих осях.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        model = self.model
        if model.median_params is None:
            raise RuntimeError("Нет медианных параметров. Запустите MCMC или загрузите пакет анализа.")

        mp = model.median_params
        cosmo = self._setup_cosmology(mp)
        arr = model._arrays

        chi2_sn = chi2_grb = None
        dof_sn = dof_grb = 0

        z_all, z_smooth = self._get_z_range(arr)
        if z_all is None:
            return None

        info = self._compute_dof_and_labels(arr, mp)
        dof_sn = info['dof_sn']
        dof_grb = info['dof_grb']

        if axes is not None:
            ax1, ax2 = axes
            fig = ax1.figure
            save_this = False
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                           gridspec_kw={'height_ratios': [3, 1]},
                                           sharex=True)
            save_this = save

        ax1.text(0.02, 0.98, ', '.join(info['parts']), transform=ax1.transAxes,
                 fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self._plot_theoretical_curves(ax1, z_smooth, cosmo)

        # Линейный закон Хаббла
        H0_linear = mp.get('H0', 70.0)
        mu_linear = 5.0 * np.log10(const.C_KM_S * z_smooth / H0_linear) + 25.0
        ax1.plot(z_smooth, mu_linear, ':', color='gray', linewidth=2,
                 label='Hubble law (linear)')

        chi2_sn = self._plot_sn_on_hubble(ax1, ax2, arr, cosmo)

        if arr.get('is_cloud', False):
            chi2_grb = self._plot_grb_clouds_on_hubble(ax1, ax2, arr, mp, cosmo)
        elif len(arr['z_grb']) > 0:
            chi2_grb = self._plot_grb_sigma_int_on_hubble(ax1, ax2, arr, mp, cosmo)

        # Общий χ²/dof в легенду нижнего графика
        if chi2_sn is not None and chi2_grb is not None:
            total_chi2 = chi2_sn + chi2_grb
            total_dof = dof_sn + dof_grb
            ax2.plot([], [], ' ', label=rf'Total $\chi^2/dof = {total_chi2:.1f}/{total_dof}$')

        # Гарантируем одинаковый масштаб по X (на всякий случай)
        ax2.set_xlim(ax1.get_xlim())

        ax1.legend(loc='lower right', fontsize=10)
        ax1.set_ylabel(r'$\mu$', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linestyle='--', lw=2)
        ax2.set_xlabel('Redshift z', fontsize=14)
        ax2.set_ylabel(r'$\Delta\mu$', fontsize=14)
        if len(arr['z_grb']) > 0:
            ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        if save_this:
            # Используем фиксированные отступы вместо tight_layout
            plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, hspace=0.25)
            fname = os.path.join(self.results_path, f'hubble_{model._mode_used}.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            logger.info(f"Диаграмма Хаббла сохранена: {fname}")

        return fig if axes is None else None, (ax1, ax2)

    # -----------------------------------------------------------------
    # Trace plot
    # -----------------------------------------------------------------
    def _trace_single_parameter(self, ax, chain, samples, param_name,
                                start, n_walkers, n_total):
        """Отрисовать один параметр trace plot."""
        steps_full = np.arange(1, n_total + 1)
        steps_post = np.arange(start + 1, n_total + 1)
        if n_walkers > 50:
            idx = np.random.choice(n_walkers, 50, replace=False)
        else:
            idx = np.arange(n_walkers)
        for j in idx:
            ax.plot(steps_full, chain[:, j], alpha=0.3, lw=0.5)
        if start > 0:
            ax.axvline(start, color='gray', ls='--', lw=1, alpha=0.7, label='Burn-in')
        med = np.median(chain[start:, :], axis=1)
        line, = ax.plot(steps_post, med, 'k-', lw=2, label='Median')
        p16, p50, p84 = _percentiles(samples)
        fmt = '.3f' if param_name == 'Ode0' else '.2f'     # ← для Ode0 три знака
        ax.legend([line], [_label_with_errors(param_name, p16, p50, p84, fmt)], fontsize=14)
        ax.set_ylabel(PARAM_LABELS.get(param_name, param_name), fontsize=16)
        ax.grid(True, alpha=0.3)

    def plot_trace(self, n_discard=None, figsize=None, save=True, axes=None):
        """Построить trace plots для всех варьируемых параметров.

        Parameters
        ----------
        n_discard : int, optional
            Число отбрасываемых шагов (по умолчанию из модели).
        figsize : tuple, optional
            Размер фигуры.
        save : bool
            Сохранить ли в файл.
        axes : array-like of matplotlib.axes.Axes, optional
            Одномерный массив осей (должен соответствовать числу параметров).

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        model = self.model
        if model.samples is None or len(model.samples) == 0:
            logger.warning("Нет семплов для построения posterior.")
            return None
        varying = model._varying_at_run
        ndim = len(varying)
        if n_discard is None:
            n_discard = model.n_discard
        start = n_discard
        n_total = model.chain.shape[0]
        if start >= n_total:
            return None
        n_walkers = model.chain.shape[1]

        if axes is not None:
            fig = axes[0].figure
            save_this = False
        else:
            if figsize is None:
                figsize = (16, 2.5 * ndim)
            fig, axes_arr = plt.subplots(ndim, 1, figsize=figsize, squeeze=False,
                                         sharex=True)  # общая ось X
            axes = axes_arr[:, 0]
            save_this = save

        fig.suptitle('Trace plots', fontsize=18)

        for i, p in enumerate(varying):
            ax = axes[i]
            ax.set_xscale('log')
            ax.set_xlim(1, 10 ** np.ceil(np.log10(n_total)))
            self._trace_single_parameter(ax, model.chain[:, :, i],
                                         model.samples[:, i], p,
                                         start, n_walkers, n_total)

        # Подпись оси X только на последнем графике
        axes[-1].set_xlabel('MCMC step', fontsize=16)

        if save_this:
            plt.tight_layout()
            fname = os.path.join(self.results_path, 'trace.png')
            plt.savefig(fname, dpi=150)
            logger.info(f"Trace сохранён: {fname}")
        return fig

    # -----------------------------------------------------------------
    # Posterior distributions
    # -----------------------------------------------------------------
    def _plot_posterior_one(self, ax, vals, param_name):
        """Построить гистограмму и split-normal для одного параметра."""
        if len(vals) == 0:
            return
        ax.hist(vals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        p16, p50, p84 = _percentiles(vals)
        for perc, c, ls in zip([p16, p50, p84], ['orange', 'red', 'orange'], ['--', '-', '--']):
            ax.axvline(perc, color=c, linestyle=ls, lw=1.5)
        sigma_l, sigma_r = p50 - p16, p84 - p50
        sn = SplitNormal(median=p50, lower_err=sigma_l, upper_err=sigma_r)
        x_min = max(vals.min(), p50 - 4 * sigma_l)
        x_max = min(vals.max(), p50 + 4 * sigma_r)
        x_grid = np.linspace(x_min, x_max, 200)
        ax.plot(x_grid, sn.pdf(x_grid), 'b-', lw=2, label='Split-normal fit')
        fmt = '.3f' if param_name == 'Ode0' else '.2f'         # ← для Ode0 три знака
        ax.axvline(p50, color='red', lw=1.5,
                   label=_label_with_errors(param_name, p16, p50, p84, fmt))
        ax.set_xlabel(PARAM_LABELS.get(param_name, param_name), fontsize=12)
        ax.legend(fontsize=10, framealpha=0.4)
        ax.grid(True, alpha=0.3)

    def plot_posterior_distributions(self, figsize=(15, 10), save=True, axes=None):
        """Построить маргинальные апостериорные распределения параметров.

        Parameters
        ----------
        figsize : tuple
            Размер фигуры.
        save : bool
            Сохранить ли в файл.
        axes : 2D array of matplotlib.axes.Axes, optional
            Двумерный массив осей (строки x столбцы).

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        model = self.model
        if model.samples is None or len(model.samples) == 0:
            logger.warning("Нет семплов для построения posterior.")
            return None
        varying = model._varying_at_run
        ndim = len(varying)
        n_cols = min(3, ndim)
        n_rows = (ndim + n_cols - 1) // n_cols
        if axes is not None:
            fig = axes.flat[0].figure if hasattr(axes, 'flat') else axes[0].figure
            save_this = False
        else:
            # Увеличена высота: 4.0 дюйма на каждую строку
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(15, 4.0 * n_rows),
                                     gridspec_kw={'hspace': 0.5})
            axes = np.atleast_2d(axes)
            save_this = save
        fig.suptitle('Posterior distributions', fontsize=16)

        for i, p in enumerate(varying):
            row, col = divmod(i, n_cols)
            self._plot_posterior_one(axes[row, col], model.samples[:, i], p)
        for i in range(ndim, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].axis('off')
        if save_this:
            plt.subplots_adjust(hspace=0.5, wspace=0.3)
            fname = os.path.join(self.results_path, 'posterior.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            logger.info(f"Posterior сохранён: {fname}")
        return fig

    # -----------------------------------------------------------------
    # Corner plot
    # -----------------------------------------------------------------
    def _compute_corner_ranges(self, varying):
        """Вычислить диапазоны для углового графика."""
        ranges = []
        for i, p in enumerate(varying):
            vals = self.model.samples[:, i]
            p50 = np.percentile(vals, 50)
            s_left = p50 - np.percentile(vals, 16)
            s_right = np.percentile(vals, 84) - p50
            w = 4 * max(s_left, s_right)
            ranges.append((p50 - w, p50 + w))
        return ranges

    def plot_corner(self, truths=None, figsize=(12, 10), save=True, axes=None):
        """Построить corner plot (попарные распределения параметров).

        Заголовки на диагоналях содержат медиану и ошибки; форматы:
        H0 и b — 1 знак после запятой, Ode0 — 3 знака, остальные — 2 знака.

        Parameters
        ----------
        truths : dict, optional
            Истинные значения параметров (если есть).
        figsize : tuple
            Размер фигуры.
        save : bool
            Сохранить ли в файл.
        axes : matplotlib.figure.Figure or array of Axes, optional
            Если передан, используется существующая фигура.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        model = self.model
        if model.samples is None or len(model.samples) == 0:
            logger.warning("Нет семплов для построения posterior.")
            return None

        varying = model._varying_at_run
        ranges = self._compute_corner_ranges(varying)
        labels = [PARAM_LABELS.get(p, p) for p in varying]
        truths_vals = [truths.get(p) if truths else None for p in varying]

        if axes is not None:
            fig = axes.figure if hasattr(axes, 'figure') else axes[0].figure
            save_this = False
        else:
            fig = plt.figure(figsize=figsize)
            save_this = save

        fig = corner.corner(model.samples, labels=labels, truths=truths_vals,
                            show_titles=False,
                            quantiles=[0.16, 0.5, 0.84],
                            fig=fig, smooth=0.9, range=ranges,
                            labelpad=0.18)

        ndim = len(varying)
        axes_array = np.array(fig.axes).reshape((ndim, ndim))

        for i, param in enumerate(varying):
            vals = model.samples[:, i]
            p16, p50, p84 = np.percentile(vals, [16, 50, 84])
            sigma_l = p50 - p16
            sigma_r = p84 - p50

            if param in ('H0', 'b'):
                fmt = '.1f'
            elif param == 'Ode0':                 # ← особый формат для Ode0
                fmt = '.3f'
            else:
                fmt = '.2f'

            param_label = PARAM_LABELS.get(param, param)
            title = f"{param_label} = ${p50:{fmt}} ^{{+{sigma_r:{fmt}}}} _{{\,-{sigma_l:{fmt}}}}$"
            axes_array[i, i].set_title(title, fontsize=10)

        for ax in fig.axes:
            ax.tick_params(labelsize=12)

        plt.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.95)

        if save_this:
            fname = os.path.join(self.results_path, 'corner.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            logger.info(f"Corner plot сохранён: {fname}")
        return fig

    # -----------------------------------------------------------------
    # Построить все графики сразу
    # -----------------------------------------------------------------
    def plot_all(self, truths=None):
        """Построить и сохранить все основные графики.

        Parameters
        ----------
        truths : dict, optional
            Истинные значения параметров для corner plot.
        """
        self.plot_trace()
        self.plot_posterior_distributions()
        self.plot_corner(truths=truths)
        self.plot_hubble_diagram()


    # -----------------------------------------------------------------
    # LaTeX-формат медианных данных
    # -----------------------------------------------------------------
    def export_median_table_latex(self, filename="median_params.tex"):
        """
        Сохранить таблицу медианных параметров в формате LaTeX.

        Таблица содержит имена параметров, медианные значения и ошибки
        (16-й и 84-й процентили). Формат чисел настраивается для Ode0
        (три знака) и остальных (два знака).

        Parameters
        ----------
        filename : str
            Имя выходного файла (будет сохранён в папку результатов).
        """
        if self.model.median_params is None:
            logger.warning("Нет медианных параметров для экспорта.")
            return

        varying = self.model._varying_at_run
        ordered = self.model._all_params_names

        lines = []
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\hline")
        lines.append(r"Параметр & Медиана & Нижняя ошибка & Верхняя ошибка \\")
        lines.append(r"\hline")

        for name in ordered:
            if name in self.model.fixed:
                val = self.model.fixed[name]
                lines.append(rf"{PARAM_LABELS.get(name, name)} & {val:.3f} & (фикс.) & (фикс.) \\")
            else:
                if name not in varying:
                    continue
                idx = varying.index(name)
                vals = self.model.samples[:, idx]
                p16, p50, p84 = np.percentile(vals, [16, 50, 84])
                fmt = '.3f' if name == 'Ode0' else '.2f'
                lines.append(
                    rf"{PARAM_LABELS.get(name, name)} & ${p50:{fmt}}$ & ${p50 - p16:{fmt}}$ & ${p84 - p50:{fmt}}$ \\"
                )

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

        tex_path = os.path.join(self.results_path, filename)
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        logger.info(f"Таблица сохранена в {tex_path}")