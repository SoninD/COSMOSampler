"""
Ядро MCMC-подгонки: байесовский анализ космологических параметров
(в параметризации H0, Ode0, w, Ok0) и корреляции Амати.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import pickle
import sys
import numpy as np
import pandas as pd
import emcee
import json
import logging
from datetime import datetime
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.cosmology import Cosmology
from core.generators import GRBCloudGenerator
from core.likelihoods import BaseLikelihood, AmatiSNLikelihood
from core.paths import ProjectPaths
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

logger = logging.getLogger(__name__)


class FitterModel:
    """
    Байесовская подгонка космологических параметров и параметров Амати.

    Позволяет добавлять данные сверхновых и гамма-всплесков,
    фиксировать/освобождать параметры, запускать MCMC (emcee),
    сохранять/загружать состояние и пакеты анализа.

    Parameters
    ----------
    results_dir : str, optional
        Имя папки (или путь) для сохранения результатов. По умолчанию ''.
    create_folder : bool, optional
        Создавать ли папку для результатов. По умолчанию True.
    use_timestamp : bool, optional
        Использовать ли подпапку с временной меткой. По умолчанию True.
    """

    def __init__(self, results_dir='', create_folder=True, use_timestamp=True):
        # Данные
        self.sn_df = None
        self.grb_df = None
        self._clouds = None

        self.fixed = {}
        self.cosmo = Cosmology()

        # Границы параметров (Ode0 вместо Om0)
        self._full_bounds = {
            'H0': (55, 75),
            'Ode0': (0.09, 0.99),
            'w': (-2.0, -0.5),
            'Ok0': (-1.0, 1.0),
            'a': (0.0, 5.0),
            'b': (35, 55),
            'k': (-4.0, 4.0),
            'sigma_int': (0.01, 10.0)
        }

        # Папка результатов
        paths = ProjectPaths()
        if create_folder:
            if use_timestamp:
                base = paths.results_dir() / results_dir if results_dir else paths.results_dir()
                self.results_path = self._create_folder(str(base))
            else:
                if results_dir:
                    self.results_path = results_dir
                    os.makedirs(self.results_path, exist_ok=True)
                else:
                    self.results_path = self._create_folder(str(paths.results_dir()))
        else:
            self.results_path = str(paths.results_dir() / results_dir) if results_dir else str(paths.results_dir())

        # Априорная функция
        self.prior = None

        # Параметры MCMC
        self.n_steps = None
        self.n_walkers = None
        self.n_discard = 0
        self.chain = None
        self.samples = None
        self.backend = None
        self.backend_file = None
        self.median_params = None
        self.mcmc_time = 0.0
        self._varying_at_run = None
        self._mode_used = None
        self._all_params_names = []
        self._all_params_default = []
        self._bounds = {}
        self._arrays = {}
        self.target_steps = None
        self.likelihood = None
        self._analysis_mode = False
        self.abort_requested = False

    # -----------------------------------------------------------------
    # Управление флагом прерывания
    # -----------------------------------------------------------------
    def request_abort(self):
        """Запросить мягкое прерывание MCMC (из GUI)."""
        self.abort_requested = True

    def reset_abort(self):
        """Сбросить флаг прерывания перед новым запуском."""
        self.abort_requested = False

    # -----------------------------------------------------------------
    # Добавление данных
    # -----------------------------------------------------------------
    def add_sn(self, df, z_upper_limit=None):
        """
        Добавить данные сверхновых.

        Parameters
        ----------
        df : pandas.DataFrame
            Таблица с колонками 'zcmb', 'mu', 'dmu'.
        z_upper_limit : float, optional
            Максимальное красное смещение для фильтрации.
        """
        if df is None:
            return
        if z_upper_limit is not None:
            df = df[df['zcmb'] < z_upper_limit].copy()
        self.sn_df = df.copy()

    def add_grb(self, df):
        """
        Добавить данные гамма-всплесков.

        Parameters
        ----------
        df : pandas.DataFrame
            Таблица с колонками 'z', 'sbolo', 'sbolo_err', 'e_pi',
            'e_pi_err_l', 'e_pi_err_u'.
        """
        if df is None:
            return
        required = {'z', 'sbolo', 'sbolo_err', 'e_pi', 'e_pi_err_l', 'e_pi_err_u'}
        if not required.issubset(df.columns):
            raise ValueError(f"GRB DataFrame должен содержать колонки: {required}")
        valid = (
            (df['sbolo'] > 0) & (df['sbolo_err'] > 0) &
            (df['e_pi'] > 0) & (df['e_pi_err_l'] > 0) & (df['e_pi_err_u'] > 0) &
            np.isfinite(df['sbolo']) & np.isfinite(df['sbolo_err']) &
            np.isfinite(df['e_pi']) & np.isfinite(df['e_pi_err_l']) & np.isfinite(df['e_pi_err_u'])
        )
        n_bad = (~valid).sum()
        if n_bad > 0:
            logger.warning(f"  Удалено {n_bad} GRB с некорректными (<=0 или inf) значениями S/E.")
            df = df[valid].copy()
        if len(df) == 0:
            logger.warning("  После фильтрации не осталось ни одного GRB.")
            return
        self.grb_df = df.copy()

    # -----------------------------------------------------------------
    # Фиксация / освобождение параметров
    # -----------------------------------------------------------------
    def fix(self, **kwargs):
        """Зафиксировать параметры (ключ=значение)."""
        for name, val in kwargs.items():
            self.fixed[name] = val

    def free(self, *names):
        """Освободить ранее зафиксированные параметры."""
        for name in names:
            self.fixed.pop(name, None)

    # -----------------------------------------------------------------
    # Установка внешнего правдоподобия / априорной функции
    # -----------------------------------------------------------------
    def set_likelihood(self, likelihood):
        """Установить пользовательский объект правдоподобия."""
        if not isinstance(likelihood, BaseLikelihood):
            raise TypeError("likelihood должен наследоваться от BaseLikelihood")
        self.likelihood = likelihood

    def set_prior(self, prior_func):
        """Установить пользовательскую функцию априорной вероятности."""
        self.prior = prior_func

    @staticmethod
    def _is_frozen():
        """Проверить, запущена ли программа из собранного PyInstaller .exe."""
        return getattr(sys, 'frozen', False)

    # -----------------------------------------------------------------
    # Запуск MCMC с нуля
    # -----------------------------------------------------------------
    def run_mcmc(self, mode='sigma_int', n_walkers=200, n_steps=2000,
                 n_discard=100, n_cloud_points=1000, cloud_seed=42):
        """
        Запустить MCMC с начальными позициями.

        Parameters
        ----------
        mode : str
            'sigma_int' или 'clouds'.
        n_walkers : int
            Число walkers.
        n_steps : int
            Число шагов.
        n_discard : int
            Число отбрасываемых шагов (burn‑in).
        n_cloud_points : int
            Размер облака точек (только для 'clouds').
        cloud_seed : int
            Seed для генератора облаков.
        """
        if self._analysis_mode:
            raise RuntimeError("Этот объект загружен из пакета анализа. Запуск MCMC невозможен.")

        self.n_discard = n_discard

        self._setup_params(mode, n_walkers, n_steps, n_cloud_points, cloud_seed)
        self._prepare_data()

        varying, fixed_vals, ndim = self._get_varying_info()
        p0 = self._initial_positions(n_walkers, ndim)

        self._init_backend(n_walkers, ndim)
        self._init_sampler(varying, fixed_vals, n_walkers)

        use_progress = not self._is_frozen()

        try:
            self.sampler.run_mcmc(p0, n_steps, progress=use_progress)
        except KeyboardInterrupt:
            logger.info("\nMCMC прерван. Цепочка автоматически сохранена в бэкенде.")
            self._finalise_after_interrupt(varying, fixed_vals, thin=20)
        else:
            self._finalise_successful(varying, fixed_vals, thin=20)

    def _get_varying_info(self):
        """Вернуть список варьируемых параметров, словарь фиксированных и размерность."""
        varying = self._varying_at_run
        fixed_vals = {p: self.fixed[p] for p in self._all_params_names if p in self.fixed}
        ndim = len(varying)
        return varying, fixed_vals, ndim

    def _initial_positions(self, n_walkers, ndim):
        """Сгенерировать начальные позиции walkers."""
        low = [self._bounds[p][0] for p in self._varying_at_run]
        high = [self._bounds[p][1] for p in self._varying_at_run]
        return np.random.uniform(low=low, high=high, size=(n_walkers, ndim))

    def _init_backend(self, n_walkers, ndim):
        """Создать HDF5‑бэкенд для сохранения цепочки."""
        self.backend_file = os.path.join(self.results_path, 'chain.h5')
        self.backend = emcee.backends.HDFBackend(self.backend_file)
        self.backend.reset(n_walkers, ndim)

    def _init_sampler(self, varying, fixed_vals, n_walkers):
        """Создать объект EnsembleSampler."""
        self.sampler = emcee.EnsembleSampler(
            n_walkers, len(varying), self._log_probability,
            args=[varying, fixed_vals, self._all_params_names, self._all_params_default],
            backend=self.backend
        )

    # -----------------------------------------------------------------
    # Возобновление прерванного MCMC
    # -----------------------------------------------------------------
    def resume_mcmc(self, extra_steps=0, thin=20):
        """
        Продолжить MCMC с того места, где он остановился.

        Parameters
        ----------
        extra_steps : int
            Дополнительные шаги сверх target_steps.
        thin : int
            Прореживание для финальной выборки.
        """
        if self._analysis_mode:
            raise RuntimeError("Этот объект загружен из пакета анализа. Запуск MCMC невозможен.")
        self._ensure_backend()
        self._restore_metadata_if_needed()

        varying, fixed_vals, _ = self._get_varying_info()
        remaining = self._calc_remaining_steps(extra_steps)
        if remaining <= 0:
            return

        logger.info(f"Продолжаем с шага {self.backend.iteration}, осталось {remaining}.")
        self._init_sampler(varying, fixed_vals, self.n_walkers)

        use_progress = not self._is_frozen()
        try:
            self.sampler.run_mcmc(None, remaining, progress=use_progress)
        except KeyboardInterrupt:
            logger.info("\nПрервано. Цепочка сохранена.")
            self._finalise_after_interrupt(varying, fixed_vals, thin=thin)
        else:
            self._finalise_successful(varying, fixed_vals, thin=thin)

    def _ensure_backend(self):
        """Убедиться, что бэкенд загружен."""
        if self.backend is None:
            backend_path = os.path.join(self.results_path, 'chain.h5')
            if not os.path.exists(backend_path):
                raise RuntimeError("Backend не найден.")
            self.backend = emcee.backends.HDFBackend(backend_path)

    def _restore_metadata_if_needed(self):
        """Загрузить метаданные, если они ещё не восстановлены."""
        meta_path = os.path.join(self.results_path, 'metadata.pkl')
        if os.path.exists(meta_path) and self._varying_at_run is None:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            for k, v in meta.items():
                setattr(self, k, v)

    def _calc_remaining_steps(self, extra_steps):
        """Вычислить оставшееся число шагов."""
        target = self.target_steps or self.backend.iteration
        return target + extra_steps - self.backend.iteration

    # -----------------------------------------------------------------
    # Загрузка сохранённого состояния
    # -----------------------------------------------------------------
    def load_state_from_backend(self, folder):
        """
        Загрузить полное состояние из папки с chain.h5 и metadata.pkl.

        Parameters
        ----------
        folder : str
            Путь к папке.
        """
        self.results_path = folder
        self.backend_file = os.path.join(folder, 'chain.h5')
        if not os.path.exists(self.backend_file):
            raise FileNotFoundError(f"Не найден {self.backend_file}")
        self.backend = emcee.backends.HDFBackend(self.backend_file)
        self._load_metadata(folder)
        self._init_likelihood()
        varying, _, _ = self._get_varying_info()
        self.sampler = emcee.EnsembleSampler(
            self.n_walkers, len(varying), self._log_probability,
            args=[varying, {p: self.fixed[p] for p in self._all_params_names if p in self.fixed},
                  self._all_params_names, self._all_params_default],
            backend=self.backend
        )
        self.chain = self.sampler.get_chain()
        self.samples = self.sampler.get_chain(discard=self.n_discard, thin=20, flat=True)
        logger.info(f"Загружено {self.chain.shape[0]} шагов из {folder}")

    def _load_metadata(self, folder):
        """Восстановить метаданные из metadata.pkl."""
        meta_path = os.path.join(folder, 'metadata.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Не найден {meta_path}")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        key_map = {
            '_varying_at_run': 'varying_names',
            '_all_params_names': 'all_params_names',
            '_all_params_default': 'all_params_default',
            '_bounds': 'bounds',
            'fixed': 'fixed',
            '_mode_used': 'mode_used',
            'target_steps': 'target_steps',
            'n_walkers': 'n_walkers',
            'n_discard': 'n_discard',
            'median_params': 'median_params',
            '_arrays': '_arrays',
            'sn_df': 'sn_df',
            'grb_df': 'grb_df',
            '_clouds': '_clouds'
        }
        for attr, key in key_map.items():
            if key in meta:
                setattr(self, attr, meta[key])
        if 'cosmo' in meta:
            self.cosmo = meta['cosmo']

    # -----------------------------------------------------------------
    # Сохранение / загрузка пакета анализа
    # -----------------------------------------------------------------
    def save_analysis_package(self, path=None):
        """Сохранить все данные, необходимые для воспроизведения графиков."""
        if path is None:
            path = os.path.join(self.results_path, 'analysis')
        os.makedirs(path, exist_ok=True)
        self._save_chain_and_samples(path)
        self._save_median_params_json(path)
        self._save_metadata_json(path)
        self._save_dataframes_and_clouds(path)
        self._save_arrays(path)
        logger.info(f"Пакет анализа сохранён в: {path}")

    def _save_chain_and_samples(self, path):
        if self.sampler is not None:
            chain = self.sampler.get_chain(discard=self.n_discard, thin=1)
            np.savez_compressed(os.path.join(path, 'chain.npz'), chain=chain)
        if self.samples is not None:
            np.savez_compressed(os.path.join(path, 'samples.npz'), samples=self.samples)

    def _save_median_params_json(self, path):
        if self.median_params is not None:
            with open(os.path.join(path, 'median_params.json'), 'w') as f:
                ser = {k: float(v) if hasattr(v, 'item') else v for k, v in self.median_params.items()}
                json.dump(ser, f, indent=2)

    def _save_metadata_json(self, path):
        meta = {
            'varying_names': self._varying_at_run,
            'all_params_names': self._all_params_names,
            'fixed': self.fixed,
            'mode_used': self._mode_used,
            'bounds': {k: list(v) for k, v in self._bounds.items()},
            'n_walkers': self.n_walkers,
            'n_steps': self.n_steps,
            'n_discard': self.n_discard,
            'target_steps': self.target_steps
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def _save_dataframes_and_clouds(self, path):
        if self.sn_df is not None:
            self.sn_df.to_csv(os.path.join(path, 'sn_data.csv'), index=False)
        if self.grb_df is not None:
            self.grb_df.to_csv(os.path.join(path, 'grb_data.csv'), index=False)
        if self._clouds is not None:
            with open(os.path.join(path, 'clouds.pkl'), 'wb') as f:
                pickle.dump(self._clouds, f)

    def _save_arrays(self, path):
        if self._arrays:
            np.savez_compressed(os.path.join(path, 'arrays.npz'), **self._arrays)

    def load_analysis_package(self, path):
        """Загрузить пакет анализа и подготовиться к визуализации."""
        self.results_path = os.path.dirname(path.rstrip('/\\')) or '.'
        self._load_metadata_json(path)
        self._load_chain_and_samples(path)
        self._load_dataframes_and_clouds(path)
        self._load_arrays(path)
        self._init_likelihood()
        self._analysis_mode = True
        logger.info(f"Данные загружены из {path}. Графики будут сохранены в {self.results_path}")

    def _load_metadata_json(self, path):
        with open(os.path.join(path, 'metadata.json')) as f:
            meta = json.load(f)
        for attr, key in [('_varying_at_run', 'varying_names'),
                          ('_all_params_names', 'all_params_names'),
                          ('_bounds', 'bounds'),
                          ('fixed', 'fixed'),
                          ('_mode_used', 'mode_used'),
                          ('target_steps', 'target_steps'),
                          ('n_walkers', 'n_walkers'),
                          ('n_steps', 'n_steps'),
                          ('n_discard', 'n_discard')]:
            setattr(self, attr, meta[key])
        self._bounds = {k: tuple(v) for k, v in meta['bounds'].items()}
        with open(os.path.join(path, 'median_params.json')) as f:
            self.median_params = json.load(f)

    def _load_chain_and_samples(self, path):
        chain_file = os.path.join(path, 'chain.npz')
        if os.path.exists(chain_file):
            self.chain = np.load(chain_file)['chain']
        samples_file = os.path.join(path, 'samples.npz')
        if os.path.exists(samples_file):
            self.samples = np.load(samples_file)['samples']

    def _load_dataframes_and_clouds(self, path):
        sn_file = os.path.join(path, 'sn_data.csv')
        self.sn_df = pd.read_csv(sn_file) if os.path.exists(sn_file) else None
        grb_file = os.path.join(path, 'grb_data.csv')
        self.grb_df = pd.read_csv(grb_file) if os.path.exists(grb_file) else None
        clouds_file = os.path.join(path, 'clouds.pkl')
        if os.path.exists(clouds_file):
            with open(clouds_file, 'rb') as f:
                self._clouds = pickle.load(f)
        else:
            self._clouds = None

    def _load_arrays(self, path):
        arrays_file = os.path.join(path, 'arrays.npz')
        if os.path.exists(arrays_file):
            arrays = dict(np.load(arrays_file, allow_pickle=True))
            for k, v in arrays.items():
                if isinstance(v, np.ndarray) and v.shape == ():
                    arrays[k] = v.item()
            self._arrays = arrays
        else:
            self._arrays = {}

    # -----------------------------------------------------------------
    # Внутренние методы подготовки параметров и данных
    # -----------------------------------------------------------------
    @staticmethod
    def _create_folder(base):
        """Создать папку с меткой времени."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(base, f'run_{ts}')
            os.makedirs(path, exist_ok=True)
            logger.info(f"Результаты в: {path}")
            return path
        except Exception:
            logger.warning("Не удалось создать папку результатов, используется '.'")
            return '.'

    def _setup_params(self, mode, n_walkers, n_steps, n_cloud_points, cloud_seed):
        """Определить имена параметров, их границы и умолчания."""
        has_sn = self.sn_df is not None
        has_grb = self.grb_df is not None
        self._validate_data(has_sn, has_grb)

        cosm_names = ['H0', 'Ode0', 'w', 'Ok0']   # <-- Ode0 вместо Om0
        amati_names = ['a', 'b', 'k'] if has_grb else []
        extra = self._select_extra_params(mode, has_grb, n_cloud_points, cloud_seed)
        all_names = cosm_names + amati_names + extra

        defaults = {'H0': 70.0, 'Ode0': 0.7, 'w': -1.0, 'Ok0': 0.0,
                    'a': 0.5, 'b': 45.0, 'k': 0.0, 'sigma_int': 1.0}

        self._all_params_names = all_names
        self._all_params_default = [defaults[n] for n in all_names]
        self._bounds = {p: self._full_bounds[p] for p in all_names}

        varying = [p for p in all_names if p not in self.fixed]
        if not varying:
            raise ValueError("Нет варьируемых параметров.")
        self._varying_at_run = varying
        self.target_steps = n_steps
        self.n_walkers = n_walkers

    def _validate_data(self, has_sn, has_grb):
        """Проверить, что достаточно данных для оценки свободных параметров."""
        if not has_sn and not has_grb:
            raise ValueError("Нет данных.")
        if not has_sn:
            free_cosm = [p for p in ['H0', 'Ode0', 'w', 'Ok0'] if p not in self.fixed]
            if free_cosm:
                raise ValueError("Без SN фиксируйте космологические параметры: " + ', '.join(free_cosm))

    def _select_extra_params(self, mode, has_grb, n_cloud_points, cloud_seed):
        """Выбрать дополнительные параметры в зависимости от режима GRB."""
        self._mode_used = mode if has_grb else None
        if not has_grb:
            logger.info("Режим: только сверхновые")
            return []
        if mode == 'sigma_int':
            logger.info("Режим: стандартный GRB с sigma_int")
            return ['sigma_int']
        elif mode == 'clouds':
            logger.info("Режим: облака точек (генерация внутри)")
            self._generate_clouds(n_cloud_points, cloud_seed)
            return []
        else:
            raise ValueError("Неизвестный режим для GRB.")

    def _generate_clouds(self, n_points, seed):
        """Сгенерировать облака точек для GRB."""
        if self.grb_df is None:
            raise RuntimeError("GRB данные не загружены.")
        gen = GRBCloudGenerator(self.grb_df)
        self._clouds = gen.generate_clouds(n_points=n_points, seed=seed)

    def _prepare_data(self):
        """Подготовить плоские массивы для быстрого доступа."""
        arr = {}
        self._prepare_sn_data(arr)
        self._prepare_grb_data(arr)
        self._arrays = arr
        self._init_likelihood()

    def _prepare_sn_data(self, arr):
        if self.sn_df is not None:
            sn = self.sn_df
            arr['z_sn'] = sn['zcmb'].values.astype(np.float64)
            arr['mu_sn'] = sn['mu'].values.astype(np.float64)
            arr['dmu_sn'] = sn['dmu'].values.astype(np.float64)
        else:
            for k in ['z_sn', 'mu_sn', 'dmu_sn']:
                arr[k] = np.array([])

    def _prepare_grb_data(self, arr):
        if self._mode_used == 'clouds' and self._clouds is not None:
            self._prepare_cloud_data(arr)
        elif self.grb_df is not None:
            self._prepare_original_grb_data(arr)
        else:
            for k in ['z_grb', 'sbolo_grb', 'sbolo_err_grb',
                      'e_pi_grb', 'e_pi_err_l_grb', 'e_pi_err_u_grb']:
                arr[k] = np.array([])
            arr['is_cloud'] = False

    def _prepare_cloud_data(self, arr):
        z_list, s_list, e_list = [], [], []
        for entry in self._clouds:
            n = len(entry['sbolo_mc'])
            z_list.append(np.full(n, entry['z']))
            s_list.append(entry['sbolo_mc'])
            e_list.append(entry['e_pi_mc'])
        arr['z_grb'] = np.concatenate(z_list).astype(np.float64)
        arr['sbolo_grb'] = np.concatenate(s_list).astype(np.float64)
        arr['e_pi_grb'] = np.concatenate(e_list).astype(np.float64)
        arr['sbolo_err_grb'] = np.zeros_like(arr['sbolo_grb'])
        arr['e_pi_err_l_grb'] = np.zeros_like(arr['e_pi_grb'])
        arr['e_pi_err_u_grb'] = np.zeros_like(arr['e_pi_grb'])
        arr['is_cloud'] = True
        arr['n_grb_orig'] = len(self.grb_df)

    def _prepare_original_grb_data(self, arr):
        grb = self.grb_df
        for col in ['z', 'sbolo', 'sbolo_err', 'e_pi', 'e_pi_err_l', 'e_pi_err_u']:
            arr[col + '_grb' if col != 'z' else 'z_grb'] = grb[col].values.astype(np.float64)
        arr['is_cloud'] = False

    def _init_likelihood(self):
        """Создать объект правдоподобия с текущими массивами."""
        self.likelihood = AmatiSNLikelihood(
            cosmo=self.cosmo,
            arrays=self._arrays,
            is_cloud=self._arrays.get('is_cloud', False)
        )

    # -----------------------------------------------------------------
    # Вычисление логарифмической вероятности
    # -----------------------------------------------------------------
    def _log_probability(self, theta, varying, fixed_vals, all_names, defaults):
        """
        Вычислить ln(правдоподобие) + ln(априорное).

        Выполняет преобразование Ode0 → Om перед обновлением космологии.
        """
        p = dict(fixed_vals)

        if self.abort_requested:
            raise KeyboardInterrupt("Прервано пользователем")

        for name, val in zip(varying, theta):
            p[name] = val
        for name in varying:
            low, high = self._bounds[name]
            if not (low <= p[name] <= high):
                return -np.inf

        # Преобразование Ode0 → Om и обновление космологии
        Ode0 = p.get('Ode0', None)
        if Ode0 is not None:
            Ok0 = p.get('Ok0', 0.0)
            Om0 = 1.0 - Ok0 - Ode0
            p['Om0'] = Om0
            self.cosmo.update(H0=p.get('H0', 70.0), Ode=Ode0, Ok=Ok0, w=p.get('w', -1.0))
        else:
            # Если Ode0 не входит в параметры (старые конфигурации), используем Om напрямую
            self.cosmo.update(H0=p.get('H0', 70.0),
                              Ode=1.0 - p.get('Om0', 0.3) - p.get('Ok0', 0.0),
                              Ok=p.get('Ok0', 0.0), w=p.get('w', -1.0))

        loglike = self.likelihood.log_probability(theta, varying, fixed_vals, all_names, defaults)
        if self.prior is not None:
            logprior = self.prior(p)
            if not np.isfinite(logprior):
                return -np.inf
            loglike += logprior
        return loglike if np.isfinite(loglike) else -np.inf

    # -----------------------------------------------------------------
    # Финализация MCMC
    # -----------------------------------------------------------------
    def _finalise_successful(self, varying, fixed_vals, thin):
        """Завершить успешный запуск MCMC."""
        self.chain = self.sampler.get_chain()
        discard = self.n_discard
        self.samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        self.n_steps = self.chain.shape[0]
        if len(self.samples) == 0:
            self._handle_empty_samples(discard)
            if len(self.samples) == 0:
                return
        self._compute_median(varying, fixed_vals)
        self._print_results(varying)
        self._save_metadata()

    def _finalise_after_interrupt(self, varying, fixed_vals, thin):
        """Завершить прерванный MCMC."""
        self.chain = self.sampler.get_chain()
        discard = self.n_discard
        self.samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        self.n_steps = self.chain.shape[0]
        if len(self.samples) == 0:
            self._handle_empty_samples(discard)
        self._compute_median(varying, fixed_vals)
        self._save_metadata()

    def _handle_empty_samples(self, discard):
        """Обработать ситуацию, когда после discarding выборка пуста."""
        logger.warning("После discarding и thinning выборка пуста. "
                       "Увеличьте n_steps или уменьшите n_discard/thin.")
        self.samples = self.sampler.get_chain(discard=discard, thin=1, flat=True)
        if len(self.samples) == 0:
            logger.error("Нет семплов даже без thinning. MCMC не дал результатов.")
        else:
            logger.info(f"Использовано thin=1 для получения {len(self.samples)} семплов.")

    def _compute_median(self, varying, fixed_vals):
        """Вычислить медианные значения варьируемых параметров."""
        medians = {}
        for i, p in enumerate(varying):
            medians[p] = float(np.median(self.samples[:, i]))
        for p, v in fixed_vals.items():
            medians[p] = v
        self.median_params = medians

    def _save_metadata(self):
        """Сохранить метаданные для возможности возобновления."""
        meta = {
            'varying_names': self._varying_at_run,
            'all_params_names': self._all_params_names,
            'all_params_default': self._all_params_default,
            'bounds': self._bounds,
            'fixed': self.fixed.copy(),
            'mode_used': self._mode_used,
            'target_steps': self.target_steps,
            'n_walkers': self.n_walkers,
            'n_discard': self.n_discard,
            'median_params': self.median_params,
            'cosmo': self.cosmo,
            '_arrays': self._arrays,
            'sn_df': self.sn_df.copy() if self.sn_df is not None else None,
            'grb_df': self.grb_df.copy() if self.grb_df is not None else None,
            '_clouds': self._clouds
        }
        with open(os.path.join(self.results_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)

    def _print_results(self, varying):
        """Вывести медианные значения параметров с ошибками."""
        if self.samples is None or len(self.samples) == 0:
            logger.warning("Нет семплов для отображения результатов.")
            return
        logger.info("=== РЕЗУЛЬТАТЫ MCMC ===")
        ordered_names = self._all_params_names
        eng_labels = {
            'H0': r'$H_0$',
            'Ode0': r'$\Omega_{de}$',
            'w': r'$w$',
            'Ok0': r'$\Omega_k$',
            'a': r'$a$',
            'b': r'$b$',
            'k': r'$k$',
            'sigma_int': r'$\sigma_{\rm int}$'
        }
        logger.info("Медианные значения параметров (16, 50, 84 процентили):\n")
        for name in ordered_names:
            if name in self.fixed:
                val = self.fixed[name]
                logger.info(f"  {eng_labels.get(name, name):15s} = {val:8.3f}  (фиксирован)")
            else:
                if name not in varying:
                    continue
                idx = varying.index(name)
                vals = self.samples[:, idx]
                p16, p50, p84 = np.percentile(vals, [16, 50, 84])
                fmt = '.3f' if name == 'Ode0' else '.2f'
                logger.info(f"  {eng_labels.get(name, name):15s} = {p50:{fmt}}  (-{p50 - p16:{fmt}} / +{p84 - p50:{fmt}})")
