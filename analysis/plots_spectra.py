"""
Модуль визуализации спектров гамма-всплесков (E²·N(E)).

Предоставляет класс `SpectralPlotter`, который на основе спектральной модели
строит графики как для отдельных GRB, так и для их совокупностей.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple, Any
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.paths import ProjectPaths
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

logger = logging.getLogger(__name__)


class SpectralPlotter:
    """
    Построение спектров вида E²·N(E) для гамма-всплесков.

    Класс предназначен для работы с реализациями спектральных моделей
    (например, CPL, Band), предоставляющих метод `get_s_e_data` и
    `spectrum`. Позволяет сохранять как общие, так и индивидуальные
    графики.

    Parameters
    ----------
    spectral_model : object
        Экземпляр модели, обладающий методами `get_s_e_data`,
        `spectrum`, а также атрибутом `cat` (каталог GRB) и
        `get_params`.
    output_dir : str or Path, optional
        Корневая директория для сохранения изображений. Если не указана,
        используется `ProjectPaths().root / 'spectra'`. Внутри неё
        создаётся поддиректория с именем класса модели (без суффикса
        'Model').

    Attributes
    ----------
    model : object
        Переданная спектральная модель.
    model_name : str
        Имя модели, очищенное от суффикса 'Model'.
    output_dir : Path
        Путь для сохранения спектров конкретной модели.

    Notes
    -----
    Перед вызовом методов необходимо убедиться, что модель инициализирована
    и содержит данные (каталог GRB загружен, параметры известны).
    """

    def __init__(self, spectral_model, output_dir=None):
        self.model = spectral_model
        # Извлекаем краткое имя модели (например, 'CPL' из 'CPLModel')
        model_class_name = type(self.model).__name__
        self.model_name = (
            model_class_name[:-5]
            if model_class_name.endswith('Model')
            else model_class_name
        )
        if output_dir is None:
            base = ProjectPaths().root / 'spectra'
        else:
            base = Path(output_dir)
        self.output_dir = base / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Спектры будут сохраняться в: %s", self.output_dir)

    # ------------------------------------------------------------------
    # Вспомогательные методы (private helpers)
    # ------------------------------------------------------------------
    def _should_save_separately(
        self,
        grb_names: Optional[Union[str, List[str]]],
        separate_files: Optional[bool]
    ) -> bool:
        """
        Определяет, нужно ли сохранять каждый GRB в отдельный файл.

        Если `separate_files` задан явно, используется его значение.
        Иначе решение принимается по количеству переданных имён:
        одно имя → отдельный файл, список из одного элемента → отдельный файл,
        иначе → общий график.

        Parameters
        ----------
        grb_names : str, list of str, or None
            Имена GRB (или одно имя) для фильтрации.
        separate_files : bool or None
            Явное указание режима отдельных файлов.

        Returns
        -------
        bool
        """
        if separate_files is not None:
            return separate_files
        if isinstance(grb_names, str):
            return True
        if isinstance(grb_names, list) and len(grb_names) == 1:
            return True
        return False

    def _get_spectra_data(self, grb_names=None):
        """
        Извлекает данные для построения спектров из модели.

        Parameters
        ----------
        grb_names : str, list of str, or None
            Имена GRB для фильтрации. Если None, возвращаются все данные.

        Returns
        -------
        pandas.DataFrame
            Таблица с колонками 'GRBname', 'z' и другими, необходимыми
            для расчёта спектра.
        """
        if grb_names is not None:
            if isinstance(grb_names, str):
                grb_names = [grb_names]
            return self.model.get_s_e_data(grb_name=grb_names)
        return self.model.get_s_e_data()

    # ------------------------------------------------------------------
    # Построение отдельных и общих графиков
    # ------------------------------------------------------------------
    def _draw_spectrum_lines(
        self,
        ax: plt.Axes,
        data,
        energy_range: Tuple[float, float]
    ) -> None:
        """
        Рисует кривые E²·N(E) для всех переданных GRB на заданной оси.

        Для каждой строки в `data` извлекается красное смещение,
        находятся параметры модели через `self.model.get_params`,
        вычисляется поток в зависимости от энергии в системе покоя,
        пересчитывается в наблюдаемую энергию и отображается кривая.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Ось, на которой будет вестись рисование.
        data : pandas.DataFrame
            Данные о GRB. Ожидаются колонки 'GRBname', 'z'.
        energy_range : tuple (E_min, E_max) in keV
            Диапазон энергии в системе покоя для расчёта кривой.
        """
        e_rest = np.logspace(
            np.log10(energy_range[0]), np.log10(energy_range[1]), 200
        )
        for _, row in data.iterrows():
            z = row['z']
            grb_row = self.model.cat[
                self.model.cat['GRBname'] == row['GRBname']
            ]
            if grb_row.empty:
                continue
            params = self.model.get_params(grb_row.iloc[0])
            e_obs = e_rest / (1 + z)
            flux = self.model.spectrum(e_rest, params)
            y_values = e_obs ** 2 * flux
            ax.plot(
                e_obs, y_values,
                label=f"{row['GRBname']} (z={z:.2f})"
            )

    def _plot_single_grb(
        self,
        grb_name: str,
        energy_range: Tuple[float, float],
        log_scale: bool,
        save: bool,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """
        Строит и, возможно, сохраняет спектр одного GRB.

        Если `ax` не передан, создаётся новая фигура. В противном случае
        график добавляется на существующую ось.

        Parameters
        ----------
        grb_name : str
            Имя GRB.
        energy_range : tuple (E_min, E_max)
            Диапазон энергии в keV.
        log_scale : bool
            Использовать ли логарифмический масштаб по обеим осям.
        save : bool
            Сохранять ли фигуру в файл. Игнорируется, если передан `ax`.
        ax : matplotlib.axes.Axes or None
            Ось для рисования. Если None, создаётся новая фигура.
        """
        data = self.model.get_s_e_data(grb_name=grb_name)
        if data.empty:
            logger.warning("Нет данных для %s.", grb_name)
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        e_rest = np.logspace(
            np.log10(energy_range[0]), np.log10(energy_range[1]), 200
        )
        row = data.iloc[0]
        z = row['z']
        grb_row = self.model.cat[
            self.model.cat['GRBname'] == grb_name
        ]
        if grb_row.empty:
            logger.warning("Строка каталога для %s не найдена.", grb_name)
            if ax is None:
                plt.close(fig)
            return
        params = self.model.get_params(grb_row.iloc[0])
        e_obs = e_rest / (1 + z)
        flux = self.model.spectrum(e_rest, params)
        y_values = e_obs**2 * flux
        ax.plot(e_obs, y_values, label=f"{grb_name} (z={z:.2f})")

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('E² × N(E) (arbitrary units)')
        ax.set_title(f'{self.model_name} Spectrum of {grb_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()
            if save:
                fname = self.output_dir / f"spectrum_{grb_name}.png"
                plt.savefig(fname, dpi=150)
                logger.info("Спектр сохранён: %s", fname)
            else:
                plt.close(fig)

    def _plot_multi_spectrum(
        self,
        data,
        energy_range: Tuple[float, float],
        log_scale: bool,
        save: bool,
        filename: str
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Строит общий график с наложенными спектрами нескольких GRB.

        Parameters
        ----------
        data : pandas.DataFrame
            Данные GRB для отображения.
        energy_range : tuple (E_min, E_max)
            Диапазон энергии в keV.
        log_scale : bool
            Логарифмический масштаб.
        save : bool
            Сохранять ли график.
        filename : str
            Имя файла для сохранения (без пути).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_spectrum_lines(ax, data, energy_range)
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('E² × N(E) (arbitrary units)')
        ax.set_title(f'{self.model_name} Spectra (E² N(E) representation)')
        ax.legend(fontsize='small', loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            fname = self.output_dir / filename
            plt.savefig(fname, dpi=150)
            logger.info("Спектры сохранены: %s", fname)
        return fig, ax

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def plot_spectra(
        self,
        grb_names: Optional[Union[str, List[str]]] = None,
        energy_range: Tuple[float, float] = (0.1, 1000),
        log_scale: bool = True,
        save: bool = True,
        filename: str = 'spectra.png',
        separate_files: Optional[bool] = None,
        ax: Optional[plt.Axes] = None
    ) -> Optional[Union[plt.Axes, Tuple[plt.Figure, plt.Axes]]]:
        """
        Основной метод для построения спектров E²·N(E) гамма-всплесков.

        В зависимости от переданных параметров может:
        - нарисовать кривые на предоставленной оси `ax`;
        - сохранить отдельные файлы для каждого GRB, если
          `separate_files=True` или передан единственный GRB;
        - создать общий график со всеми спектрами.

        Parameters
        ----------
        grb_names : str, list of str, or None, optional
            Имена целевых гамма-всплесков. Если None, используются все
            доступные в модели данные.
        energy_range : tuple (E_min, E_max), optional
            Диапазон энергии в системе покоя (keV). По умолчанию
            (0.1, 1000).
        log_scale : bool, optional
            Включить логарифмический масштаб по осям (по умолчанию True).
        save : bool, optional
            Сохранять ли график(и) в файл (по умолчанию True).
        filename : str, optional
            Имя файла для общего графика (по умолчанию 'spectra.png').
        separate_files : bool or None, optional
            Принудительно сохранить каждый GRB в отдельный файл.
            Если None, решение принимается автоматически по количеству
            переданных имён.
        ax : matplotlib.axes.Axes or None, optional
            Если передан, рисование производится на этой оси, а
            сохранение не выполняется. Возвращается переданный объект.

        Returns
        -------
        matplotlib.axes.Axes or (Figure, Axes) or None
            Если передан `ax`, возвращается он же. Для общего графика
            возвращается кортеж (fig, ax). При сохранении отдельных
            файлов возвращается None.

        Examples
        --------
        >>> plotter = SpectralPlotter(cpl_model)
        >>> plotter.plot_spectra(['GRB 970508', 'GRB 990123'],
        ...                       energy_range=(1, 1e4), separate_files=True)

        >>> fig, ax = plt.subplots()
        >>> plotter.plot_spectra(ax=ax)  # нарисовать все спектры на оси
        """
        # Рисование на внешней оси
        if ax is not None:
            data = self._get_spectra_data(grb_names)
            if data.empty:
                logger.warning("Нет данных для построения спектров.")
                return None
            self._draw_spectrum_lines(ax, data, energy_range)
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel('E² × N(E) (arbitrary units)')
            ax.set_title(
                f'{self.model_name} Spectra (E² N(E) representation)'
            )
            ax.legend(fontsize='small', loc='upper right')
            ax.grid(True, alpha=0.3)
            return ax

        # Автоматический выбор: отдельные файлы или общий график
        if self._should_save_separately(grb_names, separate_files):
            if isinstance(grb_names, str):
                grb_names = [grb_names]
            for name in grb_names:
                self._plot_single_grb(name, energy_range, log_scale, save)
            return None

        data = self._get_spectra_data(grb_names)
        if data.empty:
            logger.warning("Нет данных для построения спектров.")
            return None
        return self._plot_multi_spectrum(
            data, energy_range, log_scale, save, filename
        )