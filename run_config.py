"""
Запуск MCMC-анализа по конфигурационному JSON-файлу.

Позволяет выполнять команды ``run`` (новый запуск), ``resume``
(продолжение прерванной цепочки) и ``analyze`` (построение графиков
по сохранённым результатам) без GUI.

Примеры
-------
   python .run_config.py                  # configs/config.json
   python .run_config.py my_config.json   # другой конфиг
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.catalogue import Catalogue
from core.spectral_model import BaseSpectralModel, CPLModel
from core.fitter_model import FitterModel
from analysis.plots_result import ResultPlotter
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# ================================================================
# Вспомогательные функции
# ================================================================
def parse_fixation(fix_list):
    """
    Преобразовать список строк вида ``"ключ=значение"`` в словарь.

    Parameters
    ----------
    fix_list : list of str or None
        Список строк, например ``['Ok0=0']``.

    Returns
    -------
    fixed : dict
        Словарь фиксированных параметров {имя: значение}.
    """
    fixed = {}
    if fix_list:
        for item in fix_list:
            key, val = item.split('=')
            fixed[key] = float(val)
    return fixed


def load_json_safe(filepath):
    """
    Загрузить JSON-файл с автоопределением кодировки.

    Сначала пробует UTF-8 (с BOM), затем cp1251.

    Parameters
    ----------
    filepath : str or Path
        Путь к JSON-файлу.

    Returns
    -------
    data : dict
        Содержимое JSON.
    """
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='cp1251') as f:
            return json.load(f)


# ================================================================
# Команды
# ================================================================
def run_mode(cfg):
    """
    Выполнить новый запуск MCMC.

    Параметры
    ---------
    cfg : dict
        Словарь конфигурации, загруженный из JSON.
        Ожидаемые ключи:
        - mode (str) : 'sigma_int' или 'clouds'
        - n_walkers, n_steps, n_discard, n_cloud_points, cloud_seed
        - fix (list of str) : список фиксаций вида "par=val"
        - t90min, t90max, grb_z_min, grb_z_max, ep_min, ep_max,
          ep_err_max, sn_z_min, sn_z_max (фильтры каталогов)
        - sn_csv, grb_csv (пути к CSV, если не используются каталоги)
        - swift1, swift2, pantheon (альтернативные пути к каталогам)

    Результат
    ---------
    None
        Цепочка MCMC сохраняется в папку результатов, строятся графики.
    """
    logger.info("=== Загрузка данных ===")
    sn_csv = cfg.get('sn_csv')
    grb_csv = cfg.get('grb_csv')
    if sn_csv or grb_csv:
        logger.info("Используются готовые CSV-файлы.")
        sn_df = pd.read_csv(sn_csv) if sn_csv else None
        grb_df = pd.read_csv(grb_csv) if grb_csv else None
        if sn_df is not None:
            logger.info(f"  Сверхновых (CSV): {len(sn_df)}")
        if grb_df is not None:
            logger.info(f"  GRB (CSV): {len(grb_df)}")
    else:
        cat = Catalogue(
            grb_t90min=cfg.get('t90min', 2.0),
            grb_t90max=cfg.get('t90max', 1e3),
            grb_z_min=cfg.get('grb_z_min', 1.0),
            grb_z_max=cfg.get('grb_z_max', 10.0),
            grb_ep_min=cfg.get('ep_min', 0.0),
            grb_ep_max=cfg.get('ep_max', 1e3),
            grb_ep_err_max=cfg.get('ep_err_max', 100.0),
            sn_z_min=cfg.get('sn_z_min', 0.0),
            sn_z_max=cfg.get('sn_z_max', 2.3)
        )
        if cfg.get('swift1'):
            cat.swift1_path = cfg['swift1']
        if cfg.get('swift2'):
            cat.swift2_path = cfg['swift2']
        if cfg.get('pantheon'):
            cat.pantheon_path = cfg['pantheon']

        sn_df = cat.sn_data
        grb_df = None
        logger.info(f"Сверхновых: {len(sn_df)}")

        if not cat.grb_data.empty:
            processor = CPLModel(cat.grb_data)
            grb_df = processor.get_s_e_data()

        logger.info(f"Гамма-всплесков: {len(grb_df)}")

    model = FitterModel()
    if sn_df is not None and len(sn_df) > 0:
        model.add_sn(sn_df)
    if grb_df is not None and len(grb_df) > 0:
        model.add_grb(grb_df)

    fix_dict = parse_fixation(cfg.get('fix', []))
    model.fix(**fix_dict)
    if fix_dict:
        logger.info(f"Зафиксированы: {fix_dict}")

    logger.info("\n=== Запуск MCMC ===")
    model.run_mcmc(
        mode=cfg['mode'],
        n_walkers=cfg.get('n_walkers', 50),
        n_steps=cfg.get('n_steps', 2000),
        n_discard=cfg.get('n_discard', 500),
        n_cloud_points=cfg.get('n_cloud_points', 500),
        cloud_seed=cfg.get('cloud_seed', 42)
    )

    if model.samples is None or len(model.samples) == 0:
        logger.warning("MCMC не вернул выборку. Визуализация пропущена.")
        return

    logger.info("\n=== Сохранение результатов и графики ===")
    model.save_analysis_package()
    plotter = ResultPlotter(model)
    plotter.plot_all()
    logger.info("Готово.")


def resume_mode(cfg):
    """
    Продолжить прерванную цепочку MCMC.

    Параметры
    ---------
    cfg : dict
        Содержит ключ ``resume``, внутри которого:
        - path (str) : путь к папке с цепочкой (chain.h5)
        - extra_steps (int) : добавочные шаги сверх запланированных.
    """
    path = str(Path(cfg['resume']['path']))
    extra = cfg['resume'].get('extra_steps', 0)
    logger.info(f"Возобновление из {path} с extra_steps={extra}")
    model = FitterModel(create_folder=False)
    model.load_state_from_backend(path)
    model.resume_mcmc(extra_steps=extra)

    if model.samples is None or len(model.samples) == 0:
        logger.warning("MCMC не вернул выборку. Визуализация пропущена.")
        return

    model.save_analysis_package()
    plotter = ResultPlotter(model)
    plotter.plot_all()
    logger.info("Готово.")


def analyze_mode(cfg):
    """
    Загрузить сохранённый пакет анализа и построить графики.

    Параметры
    ---------
    cfg : dict
        Содержит ключ ``analyze`` с полем ``path`` — путь к папке
        с файлами analysis (chain.npz, samples.npz, metadata.json).
    """
    path = str(Path(cfg['analyze']['path']))
    logger.info(f"Анализ данных из {path}")
    model = FitterModel(create_folder=False)
    if not os.path.exists(os.path.join(path, 'metadata.json')):
        alt = os.path.join(path, 'analysis')
        if os.path.exists(os.path.join(alt, 'metadata.json')):
            path = alt
        else:
            raise FileNotFoundError(f"Не найден пакет анализа в {path}")
    model.load_analysis_package(path)

    if model.samples is None or len(model.samples) == 0:
        logger.warning("MCMC не вернул выборку. Визуализация пропущена.")
        return

    plotter = ResultPlotter(model)
    plotter.plot_all()
    logger.info("Готово.")


# ================================================================
# Точка входа
# ================================================================
if __name__ == '__main__':
    if getattr(sys, 'frozen', False):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            handlers=[logging.FileHandler('mcmc_run.log')]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            handlers=[logging.StreamHandler()]
        )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Запуск анализа по конфигурационному файлу"
    )
    parser.add_argument(
        'config', nargs='?', default='configs/config.json',
        help='Путь к JSON-файлу с параметрами (по умолчанию config.json)'
    )
    args = parser.parse_args()

    config = load_json_safe(args.config)

    command = config.get('command', 'run')
    if command == 'run':
        run_mode(config)
    elif command == 'resume':
        resume_mode(config)
    elif command == 'analyze':
        analyze_mode(config)
    else:
        logger.error(
            f"Неизвестная команда: {command}. "
            "Допустимые: run, resume, analyze."
        )
