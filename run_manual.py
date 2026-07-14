"""
Ручной запуск MCMC-анализа с параметрами, аналогичными config.json.

Этот скрипт эмулирует команду ``run`` из `.run_config.py`, но все
настройки заданы непосредственно в коде. Позволяет быстро провести
анализ без правки JSON-файла.

Пример использования
--------------------
   python .run_manual.py
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import sys
import logging
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.fitter_model import FitterModel
from core.catalogue import Catalogue
from core.spectral_model import CPLModel
from analysis.plots_result import ResultPlotter
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Настройка логгера для вывода в консоль
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('manual_run')


def main():
    """
    Выполнить полный цикл MCMC-анализа с фиксированными параметрами.

    Параметры соответствуют типовому конфигурационному файлу
    ``configs/config.json``:

    - Режим GRB: 'sigma_int'
    - Фильтры каталогов: T90 [2, 1000] с, z GRB [1, 10],
      Epeak [0, 1000] кэВ, отн. ошибка Epeak ≤ 100,
      z SN [0, 2.3]
    - MCMC: 30 walkers, 500 шагов, burn‑in 200
    - Фиксация: Ok0 = 0 (остальные космологические параметры свободны)
    - Параметры Амати: a, b, k, sigma_int — свободны
    """
    # ------------------------------------------------------------------
    # 1. Загрузка данных из каталогов SWIFT/Pantheon
    # ------------------------------------------------------------------
    logger.info("=== Загрузка данных ===")
    cat = Catalogue(
        grb_t90min=2.0,
        grb_t90max=1000.0,
        grb_z_min=1.0,
        grb_z_max=10.0,
        grb_ep_min=0.0,
        grb_ep_max=1000.0,
        grb_ep_err_max=100.0,
        sn_z_min=0.0,
        sn_z_max=2.3
    )

    sn_df = cat.sn_data
    logger.info("Сверхновых загружено: %d", len(sn_df))

    # Обработка GRB через спектральную модель CPL
    if not cat.grb_data.empty:
        processor = CPLModel(cat.grb_data)
        grb_df = processor.get_s_e_data()
        logger.info("GRB после обработки: %d", len(grb_df))
    else:
        grb_df = None
        logger.warning("GRB не найдены после фильтрации")

    # ------------------------------------------------------------------
    # 2. Создание модели и фиксация параметров
    # ------------------------------------------------------------------
    model = FitterModel()

    if sn_df is not None and len(sn_df) > 0:
        model.add_sn(sn_df)
    if grb_df is not None and len(grb_df) > 0:
        model.add_grb(grb_df)

    # Фиксируем кривизну (остальные параметры будут оцениваться)
    model.fix(Ok0=0.0)

    # ------------------------------------------------------------------
    # 3. Запуск MCMC
    # ------------------------------------------------------------------
    logger.info("=== Запуск MCMC (sigma_int) ===")
    model.run_mcmc(
        mode='sigma_int',
        n_walkers=30,
        n_steps=500,
        n_discard=200,
        n_cloud_points=100,     # не используется в sigma_int
        cloud_seed=42
    )

    if model.samples is None or len(model.samples) == 0:
        logger.error("Цепочка пуста. Увеличьте n_steps или уменьшите n_discard.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Результаты
    # ------------------------------------------------------------------
    print("\nМедианные параметры:")
    for name, val in model.median_params.items():
        print(f"  {name}: {val:.4f}")

    logger.info("=== Сохранение результатов и графики ===")
    model.save_analysis_package()
    plotter = ResultPlotter(model)
    plotter.plot_all()
    logger.info("Готово.")


if __name__ == '__main__':
    main()

