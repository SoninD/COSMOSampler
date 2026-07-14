"""
Десктопное GUI-приложение COSMOSampler для байесовского анализа
сверхновых и гамма-всплесков (корреляция Амати).

Запуск: python gui/app.py
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import sys
import os
import pandas as pd
import threading
import logging
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from tkinterweb import HtmlFrame
# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.fitter_model import FitterModel
from core.catalogue import Catalogue
from core.spectral_model import CPLModel
from core.cosmology import Cosmology
from analysis.plots_result import ResultPlotter
from analysis.plots_additional import AdditionalPlotter
from analysis.plots_spectra import SpectralPlotter
from gui.tools import RangeSlider, SingleSlider
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
matplotlib.use('TkAgg')
logger = logging.getLogger(__name__)



# ============================================================
# Класс стартового меню
# ============================================================
class ModeMenu:
    """
    Стартовое меню выбора режима работы.

    Отображает окно с кнопками запуска различных режимов.
    При выборе режима скрывает меню и открывает основное рабочее окно.

    Parameters
    ----------
    root : tk.Tk
        Корневое окно tkinter для меню.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("COSMOSampler – выбор режима")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        self.root.iconbitmap(self._resolve_icon_path())

        frame = ttk.Frame(root, padding=20)
        frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(frame, text="Выберите режим работы:",
                  font=('TkDefaultFont', 12, 'bold')).pack(pady=10)

        btn_style = ttk.Style()
        btn_style.configure('Mode.TButton', font=('TkDefaultFont', 11), padding=8)

        ttk.Button(frame, text="MCMC SN + GRB", style='Mode.TButton',
                   command=self.start_mcmc_mode).pack(pady=5)
        ttk.Button(frame, text="Справка", style='Mode.TButton', command=self.show_help).pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_help(self):
        """Открыть окно справки с встроенной документацией."""
        HelpWindow(self.root)

    @staticmethod
    def _resolve_icon_path():
        """Вернуть путь к иконке приложения."""
        if getattr(sys, 'frozen', False):
            base = sys._MEIPASS
        else:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(base, 'sparkle.ico')

    def start_mcmc_mode(self):
        """
        Скрыть меню и открыть основное окно MCMC.

        Создаётся новое окно Toplevel, в котором размещается
        экземпляр MCMCApp. При закрытии основного окна меню
        восстанавливается.
        """
        self.root.withdraw()
        main_window = tk.Toplevel()
        main_window.geometry('1200x800')
        main_window.title('COSMOSampler')
        MCMCApp(main_window, menu_root=self.root)
        # main_window.protocol("WM_DELETE_WINDOW", self.on_close)
        main_window.update_idletasks()

    def on_close(self):
        """Завершить приложение."""
        self.root.destroy()


class HelpWindow:
    """Окно справки с HTML-документацией."""
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Справка COSMOSampler")
        self.win.geometry("900x700")
        # Определяем путь к index.html
        if getattr(sys, 'frozen', False):
            base = sys._MEIPASS
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        help_index = os.path.join(base, 'gui', 'help', 'index.html')
        # Создаём HtmlFrame
        frame = HtmlFrame(self.win, horizontal_scrollbar="auto")
        frame.load_file(help_index)
        frame.pack(fill=tk.BOTH, expand=True)


# ============================================================
# Основной класс приложения
# ============================================================
class MCMCApp:
    """
    Главное окно приложения COSMOSampler.

    Предоставляет интерфейс для настройки данных, параметров
    MCMC, фиксаций, а также для запуска сэмплирования и
    визуализации результатов.

    Parameters
    ----------
    root : tk.Toplevel
        Родительское окно.
    menu_root : tk.Tk or None
        Ссылка на корневое окно меню (для возврата при закрытии).

    Notes
    -----
    Для корректной инициализации виджетов и данных используется
    отложенный вызов `root.after` для загрузки каталогов.
    """

    DEFAULT_FILTERS = {
        't90': (2.0, 1000.0),
        'grb_z': (1.0, 10.0),
        'ep': (0.0, 1000.0),
        'ep_err': 100.0,
        'sn_z': (0.0, 2.0),
    }

    def __init__(self, root, menu_root=None):
        self.root = root
        self.menu_root = menu_root
        self.root.title('COSMOSampler')
        self.root.geometry('1200x800')

        # --- Переменные состояния ---
        self.model = None
        self.plotter = None
        self.thread = None
        self.running = False

        # --- Параметры MCMC ---
        self.mode_var = tk.StringVar(value='sigma_int')
        self.n_walkers = tk.IntVar(value=50)
        self.n_steps = tk.IntVar(value=1000)
        self.n_discard = tk.IntVar(value=100)
        self.n_cloud_points = tk.IntVar(value=300)

        # --- Фиксации (BooleanVar, DoubleVar) ---
        self.fix_H0 = (tk.BooleanVar(value=True), tk.DoubleVar(value=70.0))
        self.fix_Ode0 = (tk.BooleanVar(value=True), tk.DoubleVar(value=0.7))
        self.fix_w = (tk.BooleanVar(value=True), tk.DoubleVar(value=-1.0))
        self.fix_Ok0 = (tk.BooleanVar(value=True), tk.DoubleVar(value=0.0))
        self.fix_a = (tk.BooleanVar(), tk.DoubleVar(value=1.0))
        self.fix_b = (tk.BooleanVar(), tk.DoubleVar(value=50.0))
        self.fix_k = (tk.BooleanVar(), tk.DoubleVar(value=0.0))
        self.fix_sigma = (tk.BooleanVar(), tk.DoubleVar(value=1.0))

        self.no_sn_var = tk.BooleanVar(value=False)
        self.no_grb_var = tk.BooleanVar(value=False)

        # --- Источник данных ---
        self.use_catalog = tk.BooleanVar(value=True)
        self.sn_csv_path = tk.StringVar()
        self.grb_csv_path = tk.StringVar()

        # --- Виджеты фильтров (инициализируются позже) ---
        self.t90_range = self.t90_low_entry = self.t90_high_entry = None
        self.grb_z_range = self.grb_z_low_entry = self.grb_z_high_entry = None
        self.ep_range = self.ep_low_entry = self.ep_high_entry = None
        self.ep_err_slider = self.ep_err_entry = None
        self.sn_z_range = self.sn_z_low_entry = self.sn_z_high_entry = None

        self.filter_widgets = []
        self.csv_widgets = []
        self.csv_buttons = []
        self.reset_buttons = []
        self.cloud_points_widget = None

        # --- Папки сохранения/загрузки ---
        self.save_dir_var = tk.StringVar(value="")
        self.load_dir_var = tk.StringVar(value="")

        # --- Дополнительные графики ---
        self.grb_names_var = tk.StringVar(value="GRB161117A")
        self.eh_cosmo_mode_var = tk.StringVar(value="median")
        self.eh_H0_var = tk.DoubleVar(value=70.0)
        self.eh_Ode0_var = tk.DoubleVar(value=0.7)
        self.eh_w_var = tk.DoubleVar(value=-1.0)
        self.eh_Ok0_var = tk.DoubleVar(value=0.0)
        self.eh_custom_widgets = []

        self.full_grb_df = None
        self.clouds = None
        self.additional_plotter = AdditionalPlotter()
        self.spectral_model = None
        self.spectral_plotter = None

        # --- Ссылки на виджеты фиксации ---
        self.fix_checkbuttons = {}
        self.fix_entries = {}

        # Построение интерфейса
        self._create_widgets()
        self.root.update_idletasks()
        # Отложенная загрузка данных, чтобы log_widget гарантированно существовал
        self.root.after(100, self._load_full_grb_data)
        self.root.after(200, self._create_spectral_model)
        self.root.iconbitmap(self._resolve_icon_path())
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    # ================================================================
    # Построение интерфейса
    # ================================================================
    def _create_widgets(self):
        """Создать все элементы GUI и связать переменные с обновлениями."""
        main_pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True)

        # Левая панель с прокруткой
        left_frame = ttk.Frame(main_pw, width=350)
        main_pw.add(left_frame, weight=0)

        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._create_data_section(scrollable_frame)
        self._create_filter_section(scrollable_frame)
        self._create_mcmc_section(scrollable_frame)
        self._create_fix_section(scrollable_frame)
        self._create_save_load_section(scrollable_frame)
        self._create_buttons(scrollable_frame)
        self._create_progress_bar(scrollable_frame)
        self._create_graph_buttons(scrollable_frame)
        self._create_extra_graph_section(scrollable_frame)
        self._create_right_panel(main_pw)

        self._setup_logging()
        self._update_filter_state()
        self._update_cloud_points_state()
        self._update_custom_cosmo_state()
        self._update_fix_state()
        self.use_catalog.trace_add('write', lambda *args: self._update_filter_state())
        self.mode_var.trace_add('write', lambda *args: self._update_cloud_points_state())
        self.eh_cosmo_mode_var.trace_add('write', lambda *args: self._update_custom_cosmo_state())
        self.no_sn_var.trace_add('write', lambda *args: self._update_fix_state())
        self.no_grb_var.trace_add('write', lambda *args: self._update_fix_state())
        self._check_existing_run()
        self.load_dir_var.trace_add('write', lambda *args: self._check_existing_run())

    # ----------------------------------------------------------------
    # Секции левой панели
    # ----------------------------------------------------------------
    def _create_data_section(self, parent):
        """Секция выбора источника данных."""
        frame = ttk.LabelFrame(parent, text="Данные", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Radiobutton(frame, text="Каталоги SWIFT/Pantheon",
                        variable=self.use_catalog, value=True).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Готовые CSV",
                        variable=self.use_catalog, value=False).grid(row=1, column=0, sticky="w")

        ttk.Label(frame, text="SN CSV:").grid(row=2, column=0, sticky="w")
        sn_entry = ttk.Entry(frame, textvariable=self.sn_csv_path)
        sn_entry.grid(row=2, column=1, sticky="ew")
        sn_btn = ttk.Button(frame, text="Обзор", command=self._browse_sn_csv)
        sn_btn.grid(row=2, column=2, padx=5)
        ttk.Label(frame, text="GRB CSV:").grid(row=3, column=0, sticky="w")
        grb_entry = ttk.Entry(frame, textvariable=self.grb_csv_path)
        grb_entry.grid(row=3, column=1, sticky="ew")
        grb_btn = ttk.Button(frame, text="Обзор", command=self._browse_grb_csv)
        grb_btn.grid(row=3, column=2, padx=5)

        self.csv_widgets = [sn_entry, grb_entry]
        self.csv_buttons = [sn_btn, grb_btn]

    def _browse_sn_csv(self):
        """Диалог выбора CSV-файла сверхновых."""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.sn_csv_path.set(filename)

    def _browse_grb_csv(self):
        """Диалог выбора CSV-файла гамма-всплесков."""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.grb_csv_path.set(filename)

    # ----------------------------------------------------------------
    # Фильтры каталогов
    # ----------------------------------------------------------------
    def _create_filter_section(self, parent):
        """Секция фильтров с двусторонними слайдерами и полями ввода."""
        frame = ttk.LabelFrame(parent, text="Фильтры каталогов", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Настройка колонок для выравнивания
        frame.grid_columnconfigure(0, minsize=180, weight=0)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, minsize=70, weight=0)
        frame.grid_columnconfigure(3, minsize=20, weight=0)
        frame.grid_columnconfigure(4, minsize=70, weight=0)
        frame.grid_columnconfigure(5, minsize=30, weight=0)

        # Переменные значений
        self.t90_low_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['t90'][0])
        self.t90_high_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['t90'][1])
        self.grb_z_low_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['grb_z'][0])
        self.grb_z_high_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['grb_z'][1])
        self.ep_low_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['ep'][0])
        self.ep_high_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['ep'][1])
        self.ep_err_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['ep_err'])
        self.sn_z_low_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['sn_z'][0])
        self.sn_z_high_var = tk.DoubleVar(value=self.DEFAULT_FILTERS['sn_z'][1])

        row = 0
        # T90
        self.t90_range = RangeSlider(frame, 0.0, 2000.0, *self.DEFAULT_FILTERS['t90'],
                                     width=180, height=30, bg='SystemButtonFace', highlightthickness=0)
        self._create_range_row(frame, row, "T90 [s]:", self.t90_range,
                               self.t90_low_var, self.t90_high_var, self.DEFAULT_FILTERS['t90'])
        self.filter_widgets.extend([self.t90_range, self.t90_low_entry, self.t90_high_entry])
        row += 1

        # z GRB
        self.grb_z_range = RangeSlider(frame, 0.1, 20.0, *self.DEFAULT_FILTERS['grb_z'],
                                       width=180, height=30, bg='SystemButtonFace', highlightthickness=0)
        self._create_range_row(frame, row, "z GRB:", self.grb_z_range,
                               self.grb_z_low_var, self.grb_z_high_var, self.DEFAULT_FILTERS['grb_z'])
        self.filter_widgets.extend([self.grb_z_range, self.grb_z_low_entry, self.grb_z_high_entry])
        row += 1

        # Epeak
        self.ep_range = RangeSlider(frame, 0.0, 2000.0, *self.DEFAULT_FILTERS['ep'],
                                    width=180, height=30, bg='SystemButtonFace', highlightthickness=0)
        self._create_range_row(frame, row, "Epeak [keV]:", self.ep_range,
                               self.ep_low_var, self.ep_high_var, self.DEFAULT_FILTERS['ep'])
        self.filter_widgets.extend([self.ep_range, self.ep_low_entry, self.ep_high_entry])
        row += 1

        # Макс. отн. ошибка Epeak
        ttk.Label(frame, text="Макс. отн. ошибка Epeak:").grid(row=row, column=0, sticky="w", padx=(5,0))
        self.ep_err_slider = SingleSlider(frame, 0.0, 200.0, self.DEFAULT_FILTERS['ep_err'],
                                          width=180, height=30, bg='SystemButtonFace', highlightthickness=0)
        self.ep_err_slider.grid(row=row, column=1, sticky="ew", padx=5)
        self.ep_err_entry = ttk.Entry(frame, textvariable=self.ep_err_var, width=7)
        self.ep_err_entry.grid(row=row, column=2, padx=5)
        ttk.Label(frame, text="").grid(row=row, column=3)
        ttk.Label(frame, text="").grid(row=row, column=4)
        reset_btn = ttk.Button(frame, text="↺", width=3,
                               command=lambda: self._reset_single(self.ep_err_slider, self.ep_err_var,
                                                                  self.DEFAULT_FILTERS['ep_err']))
        reset_btn.grid(row=row, column=5, padx=5)
        self.reset_buttons.append(reset_btn)
        self.ep_err_slider.bind('<ButtonRelease-1>', lambda e: self._single_slider_release())
        self.ep_err_entry.bind('<Return>', lambda e: self._single_entry_changed())
        self.ep_err_entry.bind('<FocusOut>', lambda e: self._single_entry_changed())
        self.ep_err_entry.bind('<KeyRelease>', lambda e: self._check_ep_err_color())
        self.filter_widgets.extend([self.ep_err_slider, self.ep_err_entry])
        row += 1

        # z SN
        self.sn_z_range = RangeSlider(frame, 0.0, 5.0, *self.DEFAULT_FILTERS['sn_z'],
                                      width=180, height=30, bg='SystemButtonFace', highlightthickness=0)
        self._create_range_row(frame, row, "z SN:", self.sn_z_range,
                               self.sn_z_low_var, self.sn_z_high_var, self.DEFAULT_FILTERS['sn_z'])
        self.filter_widgets.extend([self.sn_z_range, self.sn_z_low_entry, self.sn_z_high_entry])

    def _create_range_row(self, parent, row, label, slider, low_var, high_var, defaults):
        """Создать строку с двусторонним слайдером, полями ввода и кнопкой сброса."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(5,0))
        slider.grid(row=row, column=1, sticky="ew", padx=5)
        low_entry = ttk.Entry(parent, textvariable=low_var, width=7)
        low_entry.grid(row=row, column=2, padx=(5,0))
        ttk.Label(parent, text="–").grid(row=row, column=3)
        high_entry = ttk.Entry(parent, textvariable=high_var, width=7)
        high_entry.grid(row=row, column=4, padx=(0,5))

        reset_btn = ttk.Button(parent, text="↺", width=3,
                               command=lambda: self._reset_range(slider, low_var, high_var, defaults))
        reset_btn.grid(row=row, column=5, padx=5)
        self.reset_buttons.append(reset_btn)

        # Привязка событий
        slider.bind('<ButtonRelease-1>', lambda e: self._range_slider_release(slider, low_var, high_var, defaults))
        low_entry.bind('<FocusOut>', lambda e: self._range_entry_changed(slider, low_var, high_var, defaults))
        high_entry.bind('<FocusOut>', lambda e: self._range_entry_changed(slider, low_var, high_var, defaults))
        low_entry.bind('<Return>', lambda e: self._range_entry_changed(slider, low_var, high_var, defaults))
        high_entry.bind('<Return>', lambda e: self._range_entry_changed(slider, low_var, high_var, defaults))
        low_entry.bind('<KeyRelease>', lambda e: self._check_range_color(slider, low_var, high_var, defaults))
        high_entry.bind('<KeyRelease>', lambda e: self._check_range_color(slider, low_var, high_var, defaults))

        # Сохранение ссылок
        if label == "T90 [s]:":
            self.t90_low_entry, self.t90_high_entry = low_entry, high_entry
        elif label == "z GRB:":
            self.grb_z_low_entry, self.grb_z_high_entry = low_entry, high_entry
        elif label == "Epeak [keV]:":
            self.ep_low_entry, self.ep_high_entry = low_entry, high_entry
        elif label == "z SN:":
            self.sn_z_low_entry, self.sn_z_high_entry = low_entry, high_entry

        self._set_entry_color(low_entry, low_var.get(), defaults[0])
        self._set_entry_color(high_entry, high_var.get(), defaults[1])

    # --- Вспомогательные методы синхронизации ползунков ---
    def _range_slider_release(self, slider, low_var, high_var, defaults):
        low, high = slider.get()
        low_var.set(round(low, 3))
        high_var.set(round(high, 3))
        self._check_range_color(slider, low_var, high_var, defaults)

    def _range_entry_changed(self, slider, low_var, high_var, defaults):
        try:
            low = low_var.get()
            high = high_var.get()
        except tk.TclError:
            return
        if low < slider.min_val:
            low = slider.min_val; low_var.set(low)
        if high > slider.max_val:
            high = slider.max_val; high_var.set(high)
        slider.set(low, high)
        self._check_range_color(slider, low_var, high_var, defaults)

    def _check_range_color(self, slider, low_var, high_var, defaults):
        try:
            low = low_var.get()
            high = high_var.get()
        except tk.TclError:
            return
        self._set_entry_color(self.t90_low_entry if low_var is self.t90_low_var else
                              self.grb_z_low_entry if low_var is self.grb_z_low_var else
                              self.ep_low_entry if low_var is self.ep_low_var else
                              self.sn_z_low_entry, low, defaults[0])
        self._set_entry_color(self.t90_high_entry if high_var is self.t90_high_var else
                              self.grb_z_high_entry if high_var is self.grb_z_high_var else
                              self.ep_high_entry if high_var is self.ep_high_var else
                              self.sn_z_high_entry, high, defaults[1])

    def _check_ep_err_color(self):
        try:
            val = self.ep_err_var.get()
        except tk.TclError:
            return
        self._set_entry_color(self.ep_err_entry, val, self.DEFAULT_FILTERS['ep_err'])

    def _set_entry_color(self, entry, value, default):
        if entry is None:
            return
        entry.configure(foreground='black' if abs(value - default) < 1e-6 else 'red')

    def _reset_range(self, slider, low_var, high_var, defaults):
        low_var.set(defaults[0])
        high_var.set(defaults[1])
        slider.set(*defaults)
        self._check_range_color(slider, low_var, high_var, defaults)

    def _reset_single(self, slider, var, default):
        var.set(default)
        slider.set(default)
        self._check_ep_err_color()

    def _single_slider_release(self):
        val = self.ep_err_slider.get()
        self.ep_err_var.set(round(val, 2))
        self._check_ep_err_color()

    def _single_entry_changed(self):
        try:
            val = self.ep_err_var.get()
        except tk.TclError:
            return
        if val < 0.0: val = 0.0; self.ep_err_var.set(0.0)
        elif val > 200.0: val = 200.0; self.ep_err_var.set(200.0)
        self.ep_err_slider.set(val)
        self._check_ep_err_color()

    # ----------------------------------------------------------------
    # Секции MCMC, фиксаций и папок
    # ----------------------------------------------------------------
    def _create_mcmc_section(self, parent):
        """Секция настроек MCMC."""
        frame = ttk.LabelFrame(parent, text="MCMC", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame, text="Режим GRB:").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(frame, textvariable=self.mode_var,
                                  values=['sigma_int', 'clouds'], state='readonly')
        mode_combo.grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Walkers:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.n_walkers).grid(row=1, column=1, sticky="ew")
        ttk.Label(frame, text="Steps:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.n_steps).grid(row=2, column=1, sticky="ew")
        ttk.Label(frame, text="Burn-in:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.n_discard).grid(row=3, column=1, sticky="ew")
        ttk.Label(frame, text="Точек в облаке:").grid(row=4, column=0, sticky="w")
        cloud_entry = ttk.Entry(frame, textvariable=self.n_cloud_points)
        cloud_entry.grid(row=4, column=1, sticky="ew")
        self.cloud_points_widget = cloud_entry

    def _create_fix_section(self, parent):
        """Секция фиксации параметров."""
        frame = ttk.LabelFrame(parent, text="Фиксировать параметры", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(frame, text="Без сверхновых (no SN)",
                        variable=self.no_sn_var).grid(row=0, column=0, sticky="w", padx=10, pady=2)
        ttk.Checkbutton(frame, text="Без гамма-всплесков (no GRB)",
                        variable=self.no_grb_var).grid(row=0, column=1, sticky="w", padx=10, pady=2)

        style = ttk.Style()
        style.configure('Fix.TCheckbutton', font=('TkDefaultFont', 11))

        def add_fix_row(row, label, bool_var, double_var, tag=None):
            cb = ttk.Checkbutton(frame, text=label, variable=bool_var,
                                 style='Fix.TCheckbutton')
            cb.grid(row=row, column=0, sticky="w")
            ent = ttk.Entry(frame, textvariable=double_var, width=8,
                            font=('TkDefaultFont', 11))
            ent.grid(row=row, column=1, padx=5)
            if tag:
                self.fix_checkbuttons[tag] = cb
                self.fix_entries[tag] = ent

        add_fix_row(1, "H₀ =", self.fix_H0[0], self.fix_H0[1], 'H0')
        add_fix_row(2, "Ω_de⁰ =", self.fix_Ode0[0], self.fix_Ode0[1], 'Ode0')
        add_fix_row(3, "w =", self.fix_w[0], self.fix_w[1], 'w')
        add_fix_row(4, "Ω_k⁰ =", self.fix_Ok0[0], self.fix_Ok0[1], 'Ok0')
        add_fix_row(5, "a =", self.fix_a[0], self.fix_a[1], 'a')
        add_fix_row(6, "b =", self.fix_b[0], self.fix_b[1], 'b')
        add_fix_row(7, "k =", self.fix_k[0], self.fix_k[1], 'k')
        add_fix_row(8, "σ_int =", self.fix_sigma[0], self.fix_sigma[1], 'sigma_int')

    def _update_fix_state(self):
        """Обновить доступность чекбоксов и полей в зависимости от no_sn/no_grb."""
        if self.no_sn_var.get():
            for tag in ['H0', 'Ode0', 'w', 'Ok0']:
                var = getattr(self, f'fix_{tag}')[0]
                var.set(True)
                self.fix_checkbuttons[tag].configure(state=tk.DISABLED)
                self.fix_entries[tag].configure(state=tk.NORMAL)
        else:
            for tag in ['H0', 'Ode0', 'w', 'Ok0']:
                self.fix_checkbuttons[tag].configure(state=tk.NORMAL)
                self.fix_entries[tag].configure(state=tk.NORMAL)

        if self.no_grb_var.get():
            for tag in ['a', 'b', 'k', 'sigma_int']:
                self.fix_checkbuttons[tag].configure(state=tk.DISABLED)
                self.fix_entries[tag].configure(state=tk.DISABLED)
        else:
            for tag in ['a', 'b', 'k']:
                self.fix_checkbuttons[tag].configure(state=tk.NORMAL)
                self.fix_entries[tag].configure(state=tk.NORMAL)
            if self.mode_var.get() == 'clouds':
                self.fix_sigma[0].set(True)
                self.fix_checkbuttons['sigma_int'].configure(state=tk.DISABLED)
                self.fix_entries['sigma_int'].configure(state=tk.DISABLED)
            else:
                self.fix_checkbuttons['sigma_int'].configure(state=tk.NORMAL)
                self.fix_entries['sigma_int'].configure(state=tk.NORMAL)

    def _update_cloud_points_state(self):
        """Синхронизация поля 'точек в облаке' и блокировка sigma_int."""
        if self.cloud_points_widget:
            self.cloud_points_widget.configure(
                state=tk.NORMAL if self.mode_var.get() == 'clouds' else tk.DISABLED)
        if not self.no_grb_var.get():
            if self.mode_var.get() == 'clouds':
                self.fix_sigma[0].set(True)
                self.fix_checkbuttons['sigma_int'].configure(state=tk.DISABLED)
                self.fix_entries['sigma_int'].configure(state=tk.DISABLED)
            else:
                self.fix_checkbuttons['sigma_int'].configure(state=tk.NORMAL)
                self.fix_entries['sigma_int'].configure(state=tk.NORMAL)

    def _update_custom_cosmo_state(self):
        state = tk.NORMAL if self.eh_cosmo_mode_var.get() == 'custom' else tk.DISABLED
        for w in self.eh_custom_widgets:
            w.configure(state=state)

    def _update_filter_state(self):
        """Переключить доступность фильтров/CSV в зависимости от источника."""
        state = tk.NORMAL if self.use_catalog.get() else tk.DISABLED
        for w in self.filter_widgets:
            if hasattr(w, 'configure_state'):
                w.configure_state(state)
            else:
                w.configure(state=state)
        for w in self.csv_widgets:
            w.configure(state=tk.DISABLED if self.use_catalog.get() else tk.NORMAL)
        for btn in self.csv_buttons:
            btn.configure(state=tk.DISABLED if self.use_catalog.get() else tk.NORMAL)
        for btn in self.reset_buttons:
            btn.configure(state=state)
        if not self.use_catalog.get():
            self.spectral_model = None

    def _check_existing_run(self):
        path = self.load_dir_var.get().strip()
        if path and os.path.isdir(path) and os.path.exists(os.path.join(path, 'chain.h5')):
            self.resume_btn.configure(state=tk.NORMAL)
        else:
            self.resume_btn.configure(state=tk.DISABLED)

    # ----------------------------------------------------------------
    # Секция сохранения/загрузки, кнопки, прогресс, графики, доп.графики
    # ----------------------------------------------------------------
    def _create_save_load_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Папки", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame, text="Сохранять в:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.save_dir_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Обзор", command=self._browse_save_dir).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="Загрузить из:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.load_dir_var).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Обзор", command=self._browse_load_dir).grid(row=1, column=2, padx=5)

    def _browse_save_dir(self):
        folder = filedialog.askdirectory(initialdir=self.save_dir_var.get())
        if folder:
            self.save_dir_var.set(folder)

    def _browse_load_dir(self):
        folder = filedialog.askdirectory(initialdir=self.load_dir_var.get())
        if folder:
            self.load_dir_var.set(folder)
            self._check_existing_run()

    def _create_buttons(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=10)

        style = ttk.Style()
        style.configure('Big.TButton', font=('TkDefaultFont', 10, 'bold'), padding=6)

        self.run_btn = ttk.Button(frame, text="Запустить MCMC", command=self.start_mcmc,
                                  style='Big.TButton')
        self.run_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.resume_btn = ttk.Button(frame, text="Продолжить MCMC", command=self.resume_mcmc,
                                     state=tk.DISABLED, style='Big.TButton')
        self.resume_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.abort_btn = ttk.Button(frame, text="Прервать MCMC", command=self._on_abort,
                                    state=tk.DISABLED, style='Big.TButton')
        self.abort_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.save_btn = ttk.Button(frame, text="Сохранить анализ", command=self.save_analysis,
                                   state=tk.DISABLED, style='Big.TButton')
        self.save_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(frame, text="Выход", command=self._on_closing,
                   style='Big.TButton').pack(side=tk.RIGHT, padx=5)

    def _create_progress_bar(self, parent):
        self.progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        self.progress_label = ttk.Label(parent, text="")
        self.progress_label.pack(fill=tk.X, padx=5)

    def _create_graph_buttons(self, parent):
        frame = ttk.LabelFrame(parent, text="Графики", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Диаграмма Хаббла", command=self.show_hubble).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Trace plot", command=self.show_trace).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Corner plot", command=self.show_corner).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Posterior", command=self.show_posterior).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="Сохранить текущий", command=self.save_current_figure).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Сохранить все графики", command=self.save_all_plots).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Экспорт в LaTeX", command=self.export_latex).pack(side=tk.LEFT, padx=2)


    def _create_extra_graph_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Доп. графики", padding=5)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame, text="Имена GRB (через запятую):").grid(row=0, column=0, sticky="w", padx=2)
        ttk.Entry(frame, textvariable=self.grb_names_var, width=25).grid(row=0, column=1, columnspan=2, sticky="ew", padx=2)

        ttk.Label(frame, text="Космология T90‑EH:").grid(row=1, column=0, sticky="w", padx=2)
        cosmo_combo = ttk.Combobox(frame, textvariable=self.eh_cosmo_mode_var,
                                   values=['median', 'custom'], state='readonly', width=8)
        cosmo_combo.grid(row=1, column=1, sticky="w", padx=2)

        custom_frame = ttk.Frame(frame)
        custom_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=2, pady=2)
        ttk.Label(custom_frame, text="H0:").grid(row=0, column=0, padx=2)
        e1 = ttk.Entry(custom_frame, textvariable=self.eh_H0_var, width=6)
        e1.grid(row=0, column=1, padx=2)
        self.eh_custom_widgets.append(e1)
        ttk.Label(custom_frame, text="Ω_de:").grid(row=0, column=2, padx=2)
        e2 = ttk.Entry(custom_frame, textvariable=self.eh_Ode0_var, width=6)
        e2.grid(row=0, column=3, padx=2)
        self.eh_custom_widgets.append(e2)
        ttk.Label(custom_frame, text="w:").grid(row=0, column=4, padx=2)
        e3 = ttk.Entry(custom_frame, textvariable=self.eh_w_var, width=6)
        e3.grid(row=0, column=5, padx=2)
        self.eh_custom_widgets.append(e3)
        ttk.Label(custom_frame, text="Ωk:").grid(row=0, column=6, padx=2)
        e4 = ttk.Entry(custom_frame, textvariable=self.eh_Ok0_var, width=6)
        e4.grid(row=0, column=7, padx=2)
        self.eh_custom_widgets.append(e4)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=5)
        ttk.Button(btn_frame, text="Облака (все)",
                   command=lambda: self._plot_clouds(None)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Облака (выбранные)",
                   command=lambda: self._plot_clouds(self._get_grb_names())).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="T90 – EH",
                   command=self._plot_t90_eh).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Спектры",
                   command=self._plot_spectra).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Сохранить текущий",
                   command=self.save_extra_figure).pack(side=tk.LEFT, padx=2)

    # ----------------------------------------------------------------
    # Правая панель
    # ----------------------------------------------------------------
    def _create_right_panel(self, main_pw):
        right_frame = ttk.Frame(main_pw)
        main_pw.add(right_frame, weight=1)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Лог")
        self.log_widget = scrolledtext.ScrolledText(log_frame, state='normal', height=15)
        self._enable_text_shortcuts(self.log_widget)
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Графики")
        self.figure_canvas = None

        self.extra_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.extra_frame, text="Доп. графики")
        self.extra_figure_canvas = None
        self.extra_current_figure = None

    # ================================================================
    # Логирование
    # ================================================================
    def _setup_logging(self):
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                try:
                    if not self.text_widget.winfo_exists():
                        return
                    msg = self.format(record) + '\n'
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, msg)
                    self.text_widget.see(tk.END)
                    self.text_widget.configure(state='disabled')
                except tk.TclError:
                    pass
        handler = TextHandler(self.log_widget)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def log_message(self, msg):
        """Записать сообщение в лог GUI (безопасно)."""
        try:
            logging.info(msg)
        except tk.TclError:
            pass

    def _enable_text_shortcuts(self, widget):
        def copy(event): widget.event_generate('<<Copy>>'); return 'break'
        def cut(event): widget.event_generate('<<Cut>>'); return 'break'
        def paste(event): widget.event_generate('<<Paste>>'); return 'break'
        def select_all(event): widget.tag_add('sel', '1.0', 'end'); return 'break'
        widget.bind('<Control-KeyPress>', lambda e: copy(e) if e.keycode == 67 else
        cut(e) if e.keycode == 88 else paste(e) if e.keycode == 86 else
        select_all(e) if e.keycode == 65 else None)

    # ================================================================
    # Загрузка данных
    # ================================================================
    def load_data(self):
        """
        Загрузить данные SN и GRB из выбранного источника.

        Returns
        -------
        sn_df : pandas.DataFrame or None
        grb_df : pandas.DataFrame or None
        """
        if self.use_catalog.get():
            t90_low, t90_high = self.t90_range.get()
            z_grb_low, z_grb_high = self.grb_z_range.get()
            ep_low, ep_high = self.ep_range.get()
            sn_z_low, sn_z_high = self.sn_z_range.get()
            cat = Catalogue(
                grb_t90min=t90_low, grb_t90max=t90_high,
                grb_z_min=z_grb_low, grb_z_max=z_grb_high,
                grb_ep_min=ep_low, grb_ep_max=ep_high,
                grb_ep_err_max=self.ep_err_var.get(),
                sn_z_min=sn_z_low, sn_z_max=sn_z_high
            )
            sn_df = cat.sn_data
            grb_raw = cat.grb_data
            grb_df = CPLModel(grb_raw).get_s_e_data() if not grb_raw.empty else None
        else:
            sn_csv = self.sn_csv_path.get()
            grb_csv = self.grb_csv_path.get()
            sn_df = pd.read_csv(sn_csv) if sn_csv else None
            grb_df = pd.read_csv(grb_csv) if grb_csv else None
        return sn_df, grb_df

    def _load_full_grb_data(self):
        """Загрузить полный каталог GRB для дополнительных графиков."""
        if not self.use_catalog.get():
            return
        try:
            z_grb_low, z_grb_high = self.grb_z_range.get()
            cat = Catalogue(
                grb_t90min=0.0, grb_t90max=1e6,
                grb_z_min=z_grb_low, grb_z_max=z_grb_high,
                grb_ep_min=0.0, grb_ep_max=1e6,
                grb_ep_err_max=self.ep_err_var.get(),
                sn_z_min=0, sn_z_max=10
            )
            grb_raw = cat.grb_data
            if not grb_raw.empty:
                self.full_grb_df = CPLModel(grb_raw).get_s_e_data()
                self.log_message(f"Загружен полный каталог GRB: {len(self.full_grb_df)} объектов.")
        except Exception as e:
            self.log_message(f"Не удалось загрузить полный каталог GRB: {e}")

    def _create_spectral_model(self):
        """Создать спектральную модель CPL для всех GRB."""
        if not self.use_catalog.get():
            self.spectral_model = None
            return
        try:
            cat = Catalogue(
                grb_t90min=0.0, grb_t90max=1e6,
                grb_z_min=0.1, grb_z_max=20.0,
                grb_ep_min=0.0, grb_ep_max=1e6,
                grb_ep_err_max=self.ep_err_var.get(),
                sn_z_min=0, sn_z_max=10
            )
            grb_raw = cat.grb_data
            if not grb_raw.empty:
                self.spectral_model = CPLModel(grb_raw)
                self.log_message("Спектральная модель (CPL) создана.")
            else:
                self.spectral_model = None
        except Exception as e:
            self.log_message(f"Не удалось создать спектральную модель: {e}")
            self.spectral_model = None

    # ================================================================
    # Управление MCMC
    # ================================================================
    def start_mcmc(self):
        """
        Запустить MCMC с текущими настройками.

        Выполняет загрузку данных, создание FitterModel,
        фиксацию параметров и запуск цепочки в отдельном потоке.
        """
        if self.running:
            messagebox.showwarning("Внимание", "MCMC уже выполняется.")
            return
        try:
            sn_df, grb_df = self.load_data()
        except Exception as e:
            messagebox.showerror("Ошибка загрузки данных", str(e))
            return
        save_dir = self.save_dir_var.get().strip()
        self.model = FitterModel(results_dir=save_dir, create_folder=True, use_timestamp=False) if save_dir else FitterModel(results_dir='', create_folder=True)
        if sn_df is not None and len(sn_df) > 0 and not self.no_sn_var.get():
            self.model.add_sn(sn_df)
        if grb_df is not None and len(grb_df) > 0 and not self.no_grb_var.get():
            self.model.add_grb(grb_df)
        fix_list = [
            ('H0', self.fix_H0), ('Ode0', self.fix_Ode0), ('w', self.fix_w), ('Ok0', self.fix_Ok0),
            ('a', self.fix_a), ('b', self.fix_b), ('k', self.fix_k), ('sigma_int', self.fix_sigma)
        ]
        for name, (flag, val_var) in fix_list:
            if flag.get():
                self.model.fix(**{name: val_var.get()})
        if self.no_grb_var.get():
            for name in ['a', 'b', 'k', 'sigma_int']:
                if name not in self.model.fixed:
                    self.model.fix(**{name: {'a':1.0,'b':50.0,'k':0.0,'sigma_int':1.0}[name]})
        if self.no_sn_var.get():
            free_cosm = [p for p in ['H0', 'Ode0', 'w', 'Ok0'] if p not in self.model.fixed]
            if free_cosm:
                messagebox.showerror("Ошибка",
                                     "Без сверхновых необходимо зафиксировать все космологические параметры.\n"
                                     f"Не зафиксированы: {', '.join(free_cosm)}")
                self.running = False
                self.run_btn.configure(state=tk.NORMAL)
                return
        self.running = True
        self.model.reset_abort()
        self.run_btn.configure(state=tk.DISABLED)
        self.resume_btn.configure(state=tk.DISABLED)
        self.abort_btn.configure(state=tk.NORMAL)
        self.progress['mode'] = 'indeterminate'
        self.progress.start()
        self.progress_label['text'] = "Идёт расчёт..."
        self.log_message("=== Запуск MCMC ===")
        self.thread = threading.Thread(target=self._run_mcmc_thread, daemon=True)
        self.thread.start()
        self._poll_progress()

    def _run_mcmc_thread(self):
        try:
            self.model.run_mcmc(
                mode=self.mode_var.get(),
                n_walkers=self.n_walkers.get(),
                n_steps=self.n_steps.get(),
                n_discard=self.n_discard.get(),
                n_cloud_points=self.n_cloud_points.get(),
                cloud_seed=42
            )
        except Exception as e:
            logging.error(f"MCMC error: {e}")
        finally:
            self.running = False
            self.root.after(0, self._mcmc_finished)

    def _poll_progress(self):
        if self.running:
            self.root.after(200, self._poll_progress)
        else:
            self.progress.stop()
            self.progress['mode'] = 'determinate'
            self.progress['value'] = 100
            self.progress_label['text'] = "Готово"

    def _mcmc_finished(self):
        """Обработчик завершения MCMC."""
        self.run_btn.configure(state=tk.NORMAL)
        self.resume_btn.configure(state=tk.NORMAL if self.model and self.model.backend else tk.DISABLED)
        self.abort_btn.configure(state=tk.DISABLED)
        self.progress.stop()
        self.progress['mode'] = 'determinate'
        self.progress['value'] = 100
        self.progress_label['text'] = "Готово"
        if self.model and self.model.samples is not None and len(self.model.samples) > 0:
            self.save_btn.configure(state=tk.NORMAL)
            self.log_message("MCMC завершён. Можно строить графики.")
            self.plotter = ResultPlotter(self.model)
            self.clouds = getattr(self.model, '_clouds', None)
            if self.clouds:
                self.log_message(f"Облака загружены ({len(self.clouds)} GRB).")
        else:
            self.log_message("MCMC не дал результатов (пустая выборка).")

    def _on_abort(self):
        """Запросить прерывание MCMC с подтверждением."""
        if self.model is None or not self.running:
            messagebox.showinfo("Информация", "MCMC не запущен.")
            return
        # Спрашиваем подтверждение
        if not messagebox.askyesno(
                "Прервать MCMC",
                "Вы действительно хотите прервать расчёт?\n\n"
                "Текущая цепочка будет сохранена, и вы сможете продолжить позже."
        ):
            return
        self.model.request_abort()
        self.log_message("Отправлен запрос на прерывание...")
        self.abort_btn.configure(state=tk.DISABLED)


    def resume_mcmc(self):
        """Продолжить MCMC из сохранённого состояния."""
        path = self.load_dir_var.get().strip()
        if not path or not os.path.exists(os.path.join(path, 'chain.h5')):
            messagebox.showerror("Ошибка", "В выбранной папке нет сохранённой цепочки.")
            return
        self.model = FitterModel(create_folder=False)
        try:
            self.model.load_state_from_backend(path)
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))
            return
        self.save_dir_var.set(path)
        self.running = True
        self.model.reset_abort()
        self.run_btn.configure(state=tk.DISABLED)
        self.resume_btn.configure(state=tk.DISABLED)
        self.abort_btn.configure(state=tk.NORMAL)
        self.progress['mode'] = 'indeterminate'
        self.progress.start()
        self.progress_label['text'] = "Идёт расчёт..."
        self.log_message("=== Возобновление MCMC ===")
        thread = threading.Thread(target=self._resume_mcmc_thread, daemon=True)
        self.thread = thread
        thread.start()
        self._poll_progress()

    def _resume_mcmc_thread(self):
        try:
            self.model.resume_mcmc()
        except Exception as e:
            logging.error(f"Resume error: {e}")
        finally:
            self.running = False
            self.root.after(0, self._mcmc_finished)

    # ================================================================
    # Отображение графиков
    # ================================================================
    def _embed_figure(self, fig, target_frame='graph'):
        """Встроить matplotlib Figure в заданную вкладку GUI."""
        if target_frame == 'extra':
            frame, canvas_attr, figure_attr = self.extra_frame, 'extra_figure_canvas', 'extra_current_figure'
        else:
            frame, canvas_attr, figure_attr = self.graph_frame, 'figure_canvas', 'current_figure'
        for widget in frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        setattr(self, canvas_attr, canvas)
        setattr(self, figure_attr, fig)

    def show_hubble(self):
        """Показать диаграмму Хаббла на вкладке 'Графики'."""
        if self.plotter is None:
            messagebox.showinfo("Информация", "Сначала запустите MCMC или загрузите результаты.")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        self.plotter.plot_hubble_diagram(axes=(ax1, ax2), save=False)
        self._embed_figure(fig)

    def show_trace(self):
        """Показать trace plot."""
        if self.plotter is None:
            messagebox.showinfo("Информация", "Сначала запустите MCMC или загрузите результаты.")
            return
        ndim = len(self.model._varying_at_run)
        fig, axes = plt.subplots(ndim, 1, figsize=(12, 2 * ndim), sharex=True, squeeze=False)
        axes = axes[:, 0]
        self.plotter.plot_trace(axes=axes, save=False)
        self._embed_figure(fig)

    def show_corner(self):
        """Показать corner plot."""
        if self.plotter is None:
            messagebox.showinfo("Информация", "Сначала запустите MCMC или загрузите результаты.")
            return
        self.plotter.plot_corner(save=False)
        self._embed_figure(plt.gcf())

    def show_posterior(self):
        """Показать posterior distributions."""
        if self.plotter is None:
            messagebox.showinfo("Информация", "Сначала запустите MCMC или загрузите результаты.")
            return
        ndim = len(self.model._varying_at_run)
        ncols = min(3, ndim)
        nrows = (ndim + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6.5 * nrows),
                                 squeeze=False, gridspec_kw={'hspace':0.5,'wspace':0.3})
        self.plotter.plot_posterior_distributions(axes=axes, save=False)
        self._embed_figure(fig)

    def _save_figure_dialog(self, figure):
        """
        Диалог сохранения фигуры с выбором векторного формата.

        Предлагает PDF, SVG и PNG. По умолчанию PDF.
        Возвращает True, если сохранение прошло успешно.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not file_path:
            return False
        try:
            # Определяем формат по расширению
            ext = os.path.splitext(file_path)[1].lower()
            fmt = 'pdf' if ext == '.pdf' else 'svg' if ext == '.svg' else 'png'
            figure.savefig(file_path, dpi=150, format=fmt)
            self.log_message(f"График сохранён в {file_path}")
            return True
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))
            return False

    def save_current_figure(self):
        """Сохранить текущий график из основной вкладки."""
        if self.figure_canvas is None or self.current_figure is None:
            messagebox.showinfo("Информация", "Нет активного графика для сохранения.")
            return
        self._save_figure_dialog(self.current_figure)

    def save_extra_figure(self):
        """Сохранить текущий дополнительный график."""
        if self.extra_figure_canvas is None or self.extra_current_figure is None:
            messagebox.showinfo("Информация", "Нет активного графика для сохранения.")
            return
        self._save_figure_dialog(self.extra_current_figure)

    def save_all_plots(self):
        """Сохранить все стандартные графики в папку результатов."""
        if self.plotter is None:
            messagebox.showinfo("Информация", "Сначала запустите MCMC или загрузите результаты.")
            return
        try:
            self.plotter.plot_all()
            self.log_message("Все графики сохранены в папку результатов.")
            messagebox.showinfo("Готово", f"Графики сохранены в {self.model.results_path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))


    def export_latex(self):
        """Экспортировать таблицу медианных параметров в LaTeX."""
        if self.plotter is None:
            messagebox.showinfo("Информация", "Сначала запустите MCMC или загрузите результаты.")
            return
        try:
            self.plotter.export_median_table_latex()
            self.log_message("Таблица экспортирована в median_params.tex в папку результатов.")
        except Exception as e:
            messagebox.showerror("Ошибка экспорта", str(e))

    # ================================================================
    # Иконка, сохранение анализа, закрытие
    # ================================================================
    @staticmethod
    def _resolve_icon_path():
        if getattr(sys, 'frozen', False):
            base = sys._MEIPASS
        else:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(base, 'sparkle.ico')

    def save_analysis(self):
        """Сохранить пакет анализа (сэмплы, метаданные)."""
        if self.model:
            try:
                self.model.save_analysis_package()
                messagebox.showinfo("Сохранение", "Пакет анализа сохранён.")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _on_closing(self):
        """
        Обработчик закрытия окна с предупреждением и остановкой MCMC.

        Если расчёт ещё идёт, отправляет запрос на прерывание и
        дожидается остановки потока перед закрытием.
        """
        # Проверяем, нужно ли предупреждение
        need_warning = (
                self.model is not None and
                (self.running or (self.model.samples is not None and len(self.model.samples) > 0))
        )
        if need_warning:
            save_path = getattr(self.model, 'results_path', 'неизвестно')
            msg = (
                "Есть несохранённые результаты MCMC.\n"
                f"При закрытии метаданные и цепочка будут записаны в:\n{save_path}\n\n"
                "Закрыть программу?"
            )
            if not messagebox.askyesno("Подтверждение закрытия", msg):
                return  # отмена

        # Отправляем запрос на прерывание, если MCMC активен
        if self.model is not None and self.running:
            self.model.request_abort()
            self.log_message("Ожидание остановки MCMC...")
            # Даём потоку время завершиться (до 2 секунд)
            self.root.after(100, self._check_mcmc_stopped_and_close)
            return

        # Если MCMC не запущен или уже остановлен – сохраняем и закрываем
        self._save_and_close()

    def _check_mcmc_stopped_and_close(self):
        """Периодически проверяет, остановился ли MCMC, и закрывает окно."""
        if self.running:
            self.root.after(100, self._check_mcmc_stopped_and_close)  # ждём ещё
        else:
            self._save_and_close()

    def _save_and_close(self):
        """Сохраняет метаданные и закрывает окно."""
        try:
            if self.model:
                self.model._save_metadata()
                logging.info("Метаданные сохранены при закрытии.")
        except Exception:
            pass
        finally:
            self.root.destroy()
            if self.menu_root:
                self.menu_root.deiconify()

        # Закрываем окно
        self.root.destroy()
        if self.menu_root:
            self.menu_root.deiconify()


    # ================================================================
    # Дополнительные графики
    # ================================================================
    def _get_grb_names(self):
        """Извлечь список имён GRB из текстового поля."""
        raw = self.grb_names_var.get().strip()
        if not raw:
            return None
        return [name.strip() for name in raw.split(',') if name.strip()]

    def _plot_clouds(self, grb_names):
        """Построить облака точек MC."""
        if self.clouds is None:
            messagebox.showinfo("Информация",
                                "Облака ещё не сгенерированы.\nЗапустите MCMC в режиме 'clouds'.")
            return
        try:
            result = self.additional_plotter.plot_clouds(self.clouds, grb_names=grb_names, save=False)
            if result is not None:
                fig = result[0] if isinstance(result, tuple) else result
                self._embed_figure(fig, target_frame='extra')
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _plot_t90_eh(self):
        """Построить диаграмму T90–EH."""
        if self.full_grb_df is None:
            messagebox.showinfo("Информация", "Полный каталог GRB не загружен.")
            return
        try:
            if self.eh_cosmo_mode_var.get() == 'median':
                if self.model is None or self.model.cosmo is None:
                    messagebox.showinfo("Информация", "Медианная космология недоступна.")
                    return
                cosmo = self.model.cosmo
            else:
                cosmo = Cosmology()
                cosmo.update(H0=self.eh_H0_var.get(), Ode=self.eh_Ode0_var.get(),
                             w=self.eh_w_var.get(), Ok=self.eh_Ok0_var.get())
            fig, ax = self.additional_plotter.plot_t90_eh(self.full_grb_df, cosmo, save=False)
            self._embed_figure(fig, target_frame='extra')
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _plot_spectra(self):
        """Построить спектры (E²·N(E)) для выбранных GRB."""
        if self.spectral_model is None:
            messagebox.showinfo("Информация", "Спектральная модель недоступна.")
            return
        if self.spectral_plotter is None:
            self.spectral_plotter = SpectralPlotter(self.spectral_model)
        grb_names = self._get_grb_names()
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.spectral_plotter.plot_spectra(grb_names=grb_names, ax=ax, save=False)
            if not ax.get_lines():
                plt.close(fig)
                messagebox.showinfo("Информация", "Для указанных имён спектры не найдены.")
                return
            self._embed_figure(fig, target_frame='extra')
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == '__main__':
    root = tk.Tk()
    ModeMenu(root)
    root.mainloop()