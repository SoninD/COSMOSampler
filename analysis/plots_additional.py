"""
Дополнительные графики: облака точек MC, диаграмма T90–EH.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
import matplotlib.pyplot as plt
import logging
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.distributions import SplitNormal
from core.paths import ProjectPaths
from core import constants as const
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
logger = logging.getLogger(__name__)


class AdditionalPlotter:
    """
    Построение дополнительных графиков, не связанных напрямую с результатами MCMC.

    Позволяет визуализировать облака точек, сгенерированные для GRB,
    а также строить диаграмму T90–EH.

    Attributes
    ----------
    output_dir : pathlib.Path
        Папка для сохранения графиков (по умолчанию 'additional_plots').
    """

    def __init__(self):
        """Инициализировать плоттер и создать папку для сохранения."""
        self.output_dir = ProjectPaths().root / 'additional_plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Внутренние методы
    # -----------------------------------------------------------------
    def _get_split_normal_params(self, entry):
        """
        Извлечь параметры SplitNormal для одного GRB.

        Parameters
        ----------
        entry : dict
            Словарь с ключами 'sbolo', 'sbolo_err', 'e_pi', 'e_pi_err_l', 'e_pi_err_u'.

        Returns
        -------
        dist_s : SplitNormal
            Распределение для S_bolo в log-пространстве.
        dist_e : SplitNormal
            Распределение для E_pi в log-пространстве.
        sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u : float
            Исходные значения.
        """
        sbolo = entry['sbolo']
        sbolo_err = entry['sbolo_err']
        e_pi = entry['e_pi']
        e_pi_err_l = entry['e_pi_err_l']
        e_pi_err_u = entry['e_pi_err_u']

        log_s_med = np.log10(sbolo)
        log_s_err_left = log_s_med - np.log10(sbolo - sbolo_err)
        log_s_err_right = np.log10(sbolo + sbolo_err) - log_s_med

        log_e_med = np.log10(e_pi)
        log_e_err_left = log_e_med - np.log10(e_pi - e_pi_err_l)
        log_e_err_right = np.log10(e_pi + e_pi_err_u) - log_e_med

        dist_s = SplitNormal(median=log_s_med, lower_err=log_s_err_left, upper_err=log_s_err_right)
        dist_e = SplitNormal(median=log_e_med, lower_err=log_e_err_left, upper_err=log_e_err_right)
        return dist_s, dist_e, sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u

    def _plot_histogram_with_pdf(self, ax, values, split_normal, orientation='vertical'):
        """
        Нарисовать гистограмму и теоретическую PDF SplitNormal.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Ось для рисования.
        values : ndarray
            Массив значений для гистограммы.
        split_normal : SplitNormal
            Объект распределения.
        orientation : str, optional
            'vertical' или 'horizontal'.
        """
        vmin, vmax = values.min(), values.max()
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 50)
        ax.hist(values, bins=bins, density=True, color='gray', alpha=0.5,
                orientation=orientation)
        grid = np.logspace(np.log10(vmin), np.log10(vmax), 200)
        pdf = split_normal.pdf(np.log10(grid)) / (grid * np.log(10))
        if orientation == 'vertical':
            ax.plot(grid, pdf, 'b-', lw=1.5)
        else:
            ax.plot(pdf, grid, 'b-', lw=1.5)

    # -----------------------------------------------------------------
    # Публичные методы построения
    # -----------------------------------------------------------------
    def plot_clouds(self, clouds, grb_names=None, log_scale=True,
                    save=False, save_path=None, ax=None, axes_dict=None):
        """
        Универсальная отрисовка облаков точек.

        Parameters
        ----------
        clouds : list of dict
            Список словарей с полями 'GRBname', 'sbolo_mc', 'e_pi_mc', а также
            исходными 'sbolo', 'sbolo_err', 'e_pi', 'e_pi_err_l', 'e_pi_err_u'.
        grb_names : str, list of str or None, optional
            Имена GRB для отображения. Если None – все GRB серыми облаками.
            Если одно имя – детальный график с гистограммами.
            Если несколько – выбранные GRB цветными облаками на фоне всех.
        log_scale : bool, optional
            Использовать логарифмический масштаб осей.
        save : bool, optional
            Сохранить ли график в файл.
        save_path : str or Path, optional
            Путь для сохранения (по умолчанию в self.output_dir).
        ax : matplotlib.axes.Axes, optional
            Ось для рисования (только для режимов 'all' и 'selected').
        axes_dict : dict, optional
            Словарь с ключами 'main', 'top', 'right' для режима одного GRB.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
        """
        if not clouds:
            raise ValueError("Облака не переданы.")

        if grb_names is None:
            return self._plot_all_grbs(clouds, log_scale, save, save_path, ax)
        elif isinstance(grb_names, str):
            return self._plot_single_grb(clouds, grb_names, log_scale, save, save_path, axes_dict)
        elif isinstance(grb_names, list) and len(grb_names) > 0:
            if len(grb_names) == 1:
                return self._plot_single_grb(clouds, grb_names[0], log_scale, save, save_path, axes_dict)
            else:
                return self._plot_selected_grbs(clouds, grb_names, log_scale, save, save_path, ax)
        else:
            raise ValueError("grb_names должен быть строкой, списком строк или None.")

    def _plot_all_grbs(self, clouds, log_scale, save, save_path, ax):
        """Нарисовать все GRB серыми облаками."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            user_ax = False
        else:
            fig = ax.figure
            user_ax = True

        first_cloud = True
        n_points = len(clouds[0]['sbolo_mc'])
        for entry in clouds:
            lbl = f'MC clouds (N={n_points})' if first_cloud else None
            ax.scatter(entry['sbolo_mc'], entry['e_pi_mc'], s=1, alpha=0.1,
                       color='gray', label=lbl)
            first_cloud = False

        first = True
        for entry in clouds:
            lbl = 'original_grb' if first else None
            ax.errorbar(entry['sbolo'], entry['e_pi'],
                        xerr=entry['sbolo_err'],
                        yerr=[[entry['e_pi_err_l']], [entry['e_pi_err_u']]],
                        fmt='o', color='red', capsize=3, label=lbl)
            first = False

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('S_bolo')
        ax.set_ylabel('E_pi')
        ax.set_title('All GRBs: clouds & observed errors')
        ax.legend()
        if not user_ax:
            plt.tight_layout()
            if save:
                fname = save_path or str(self.output_dir / 'clouds_all_grbs.png')
                plt.savefig(fname, dpi=150)
                logger.info(f"График облаков сохранён: {fname}")
        return fig, ax

    def _plot_selected_grbs(self, clouds, grb_names, log_scale, save, save_path, ax):
        """Нарисовать выбранные GRB цветами на фоне всех серых точек."""
        selected = [e for e in clouds if e['GRBname'] in grb_names]
        if not selected:
            raise ValueError("Ни один из указанных GRB не найден в облаках.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            user_ax = False
        else:
            fig = ax.figure
            user_ax = True

        for entry in clouds:
            ax.errorbar(entry['sbolo'], entry['e_pi'],
                        xerr=entry['sbolo_err'],
                        yerr=[[entry['e_pi_err_l']], [entry['e_pi_err_u']]],
                        fmt='o', color='lightgray', capsize=2, alpha=0.6,
                        markersize=2, elinewidth=0.5)

        colors = plt.cm.tab10.colors
        for i, entry in enumerate(selected):
            color = colors[i % len(colors)]
            ax.scatter(entry['sbolo_mc'], entry['e_pi_mc'], s=1, alpha=0.15, color=color)
            ax.errorbar(entry['sbolo'], entry['e_pi'],
                        xerr=entry['sbolo_err'],
                        yerr=[[entry['e_pi_err_l']], [entry['e_pi_err_u']]],
                        fmt='o', color=color, capsize=4, markersize=4, elinewidth=1.2,
                        label=entry['GRBname'])

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('S_bolo')
        ax.set_ylabel('E_pi')
        ax.set_title('Selected GRBs (clouds) with all original data')
        ax.legend()
        if not user_ax:
            plt.tight_layout()
            if save:
                fname = save_path or str(self.output_dir / 'clouds_selected.png')
                plt.savefig(fname, dpi=150)
                logger.info(f"График выбранных GRB сохранён: {fname}")
        return fig, ax

    def _plot_single_grb(self, clouds, grb_name, log_scale, save, save_path, axes):
        """Детальный график одного GRB: scatter + гистограммы + PDF."""
        entry = next((e for e in clouds if e['GRBname'] == grb_name), None)
        if entry is None:
            raise ValueError(f"GRB {grb_name} не найден в облаках.")

        dist_s, dist_e, sbolo, sbolo_err, e_pi, e_pi_err_l, e_pi_err_u = self._get_split_normal_params(entry)

        if axes is not None:
            ax_main, ax_top, ax_right = axes['main'], axes['top'], axes['right']
            fig = ax_main.figure
        else:
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                  left=0.1, right=0.9, bottom=0.1, top=0.9,
                                  wspace=0.05, hspace=0.05)
            ax_main = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        n_points = len(entry['sbolo_mc'])
        ax_main.scatter(entry['sbolo_mc'], entry['e_pi_mc'], s=1, alpha=0.1,
                        color='gray', label=f'MC (N={n_points})')
        ax_main.errorbar(sbolo, e_pi,
                         xerr=sbolo_err,
                         yerr=[[e_pi_err_l], [e_pi_err_u]],
                         fmt='o', color='red', capsize=5, label='Observed')

        if log_scale:
            ax_main.set_xscale('log')
            ax_main.set_yscale('log')
        ax_main.set_xlabel('S_bolo')
        ax_main.set_ylabel('E_pi')
        ax_main.legend()
        ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else ''))
        ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$' if y > 0 else ''))

        self._plot_histogram_with_pdf(ax_top, entry['sbolo_mc'], dist_s, 'vertical')
        self._plot_histogram_with_pdf(ax_right, entry['e_pi_mc'], dist_e, 'horizontal')

        for val, ls in zip([sbolo, sbolo - sbolo_err, sbolo + sbolo_err], ['-', '--', '--']):
            ax_top.axvline(val, color='k', linestyle=ls, linewidth=0.8)
            ax_main.axvline(val, color='k', linestyle=ls, linewidth=0.5, alpha=0.5)
        for val, ls in zip([e_pi, e_pi - e_pi_err_l, e_pi + e_pi_err_u], ['-', '--', '--']):
            ax_right.axhline(val, color='k', linestyle=ls, linewidth=0.8)
            ax_main.axhline(val, color='k', linestyle=ls, linewidth=0.5, alpha=0.5)

        ax_main.set_title(f'GRB {grb_name}')
        if axes is None:
            plt.tight_layout()
            if save:
                fname = save_path or str(self.output_dir / f'cloud_{grb_name}.png')
                plt.savefig(fname, dpi=150)
                logger.info(f"График GRB {grb_name} сохранён: {fname}")
        return fig, (ax_main, ax_top, ax_right)

    # -----------------------------------------------------------------
    # Диаграмма T90–EH
    # -----------------------------------------------------------------
    def plot_t90_eh(self, grb_df, cosmo, log_scale=True, save=False, save_path=None, ax=None):
        """
        Построить диаграмму T90 – EH.

        Parameters
        ----------
        grb_df : pandas.DataFrame
            Таблица с колонками 'z', 'T90', 'sbolo', 'e_pi'.
        cosmo : Cosmology
            Космологическая модель.
        log_scale : bool, optional
            Логарифмический масштаб.
        save : bool, optional
            Сохранить ли график.
        save_path : str, optional
            Путь для сохранения.
        ax : matplotlib.axes.Axes, optional
            Ось для рисования.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        z = grb_df['z'].values.astype(np.float64)
        sbolo = grb_df['sbolo'].values.astype(np.float64)
        e_pi = grb_df['e_pi'].values.astype(np.float64)
        t90 = grb_df['T90'].values.astype(np.float64)

        d_E_mpc = cosmo.d_E(z)
        d_E_cm = d_E_mpc * const.MPC_IN_CM
        fluence = sbolo * t90
        E_iso = 4.0 * np.pi * d_E_cm**2 * fluence / (1.0 + z)
        E_iso_51 = E_iso / 1e51

        mask = (E_iso_51 > 0) & (e_pi > 0)
        E_iso_51, e_pi, t90, z = E_iso_51[mask], e_pi[mask], t90[mask], z[mask]
        EH = (e_pi / 100.0) / (E_iso_51 ** 0.4)
        valid = np.isfinite(EH) & np.isfinite(t90) & (t90 > 0)
        t90, EH, z = t90[valid], EH[valid], z[valid]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            user_ax = False
        else:
            fig = ax.figure
            user_ax = True

        sc = ax.scatter(t90, EH, c=z, cmap='viridis', alpha=0.8, edgecolors='k', linewidth=0.3)
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('T90 [s]')
        ax.set_ylabel('EH [ (E_pi/100 keV) / (E_iso/10^51 erg)^0.4 ]')
        ax.set_title('T90 – EH diagram')
        fig.colorbar(sc, ax=ax, label='z')
        ax.grid(True, alpha=0.3)

        if not user_ax:
            plt.tight_layout()
            if save:
                fname = save_path or str(self.output_dir / 't90_eh.png')
                plt.savefig(fname, dpi=150)
                logger.info(f"Диаграмма T90-EH сохранена: {fname}")
        return fig, ax