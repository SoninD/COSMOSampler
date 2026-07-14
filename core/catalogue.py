#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from pathlib import Path

import pandas as pd
import numpy as np
import re
import sys

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from core.paths import ProjectPaths
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Catalogue:
    """
    SWIFT tables can be found at:
    SWIFT1: https://swift.gsfc.nasa.gov/results/batgrbcat/index_tables.html
    SWIFT2: https://swift.gsfc.nasa.gov/archive/grb_table/
    Pantheon table can be found at:
    PANTHEON: ?
    """

    def __init__(self,
                 grb_t90min=None, grb_t90max=None,
                 grb_z_min=None, grb_z_max=None,
                 grb_ep_min=None, grb_ep_max=None,
                 grb_ep_err_max=None,
                 sn_z_min=None, sn_z_max=None):

        # Определяем базовую папку в зависимости от способа запуска
        if getattr(sys, 'frozen', False):
            # Запущено из .exe — берём временную папку PyInstaller
            base_dir = Path(sys._MEIPASS)
        else:
            # Обычный запуск — используем ProjectPaths (корень проекта)
            base_dir = ProjectPaths().root

        self.swift1_path = str(base_dir / 'data' / 'SWIFT1.txt')
        self.swift2_path = str(base_dir / 'data' / 'SWIFT2.txt')
        self.pantheon_path = str(base_dir / 'data' / 'Pantheon.dat')

        # Дефолтные ограничения на параметры
        self.grb_t90min = grb_t90min or 0
        self.grb_t90max = grb_t90max or 1e3
        self.grb_z_min = grb_z_min or 0
        self.grb_z_max = grb_z_max or 1e4
        self.grb_ep_min = grb_ep_min or 0
        self.grb_ep_max = grb_ep_max or 1e5
        self.grb_ep_err_max = grb_ep_err_max or 100  # Максимальная относительная ошибка

        self.sn_z_min = sn_z_min or 0
        self.sn_z_max = sn_z_max or 1e4


        self.grb_data = self._get_swift()
        self.sn_data = self._get_pantheon()

    @staticmethod
    def replace_missing_values(data: pd.DataFrame, with_what, *col_names):
        """ Замена пропущенных значений в каталоге """
        if with_what == 'median':
            for col in col_names:
                data[col] = data[col].fillna(data[col].median())
        elif with_what == 'mean':
            for col in col_names:
                data[col] = data[col].fillna(data[col].mean())
        elif with_what == 'mode':
            for col in col_names:
                data[col] = data[col].fillna(data[col].mode()[0])
        return data

    def _read_swift1(self):
        """ Делаем датафрейм из таблицы SWIFT2 """
        with open(self.swift1_path, 'r') as file:
            lines = file.readlines()
            names = re.split(r'\s*\|\s*', lines[24].replace(r'#', '').strip())
            data = []
            for line in lines[25:]:
                line = line.strip()
                line0 = re.split(r'\s*\|\s*', line)
                data.append(line0)

            swift1_df_full = pd.DataFrame(data, columns=names)
            swift1_df = swift1_df_full[[names[i] for i in [0, 2, 3, 4, 5, 6, 7, 8, 15]]]

            return swift1_df

    def _read_swift2(self):
        """ Делаем датафрейм из таблицы SWIFT2 """
        name = []
        t90 = []
        s_obs = []
        s_obs_err = []
        redshift = []
        with open(self.swift2_path, 'r') as file:
            lines = file.readlines()[2:]
            for line in lines:
                line0 = line.split()
                name.append("GRB" + line0[0])
                t90.append(line0[1])
                s_obs.append((line0[2]))
                s_obs_err.append((line0[3]))
                redshift.append((line0[4]))

            swift2_df = pd.DataFrame({'GRBname': name,
                                      'T90': t90,
                                      'S_obs': s_obs,
                                      'S_obs_err': s_obs_err,
                                      'redshift': redshift})
        return swift2_df

    def _get_swift(self):
        """
        :return: Возвращаем большой датафрейм каталога SWIFT. Epeak, Epeak+, Epeak- --- пиковое значение энергии и его
        относительные ошибки сверху и снизу. DEpeak_hi, DEpeak_low --- абсолютные ошибки пиковой энергии сверху и снизу,
        для параметра альфа все аналогично. Доверительный интервал для значений DEpeak и Dalpha --- 90%.
        """

        # Считываем строки из таблиц в датафреймы
        sw1 = self._read_swift1()
        sw2 = self._read_swift2()
        # Объединяем датафреймы по имени (вхождения только в один из датафреймов выбрасываются)
        merged_sw = pd.merge(sw1, sw2, on='GRBname', how='inner')

        cols = merged_sw.columns
        for col in cols[1:]:
            merged_sw[col] = merged_sw[col].str.replace(r'(-?\d+\.\d+)-(\d+)', r'\1e-\2', regex=True)
            merged_sw[col] = merged_sw[col].str.replace(r'(-?\d+\.\d+)\+(\d+)', r'\1e+\2', regex=True)
            merged_sw[col] = pd.to_numeric(merged_sw[col], errors='coerce')
            merged_sw[col] = merged_sw[col].replace(0, np.nan)

        # Относительные ошибки определения параметра CPL альфы
        merged_sw['alpha+'] = (merged_sw['alpha_hi'] - merged_sw['alpha']) / merged_sw['alpha']
        merged_sw['alpha-'] = (merged_sw['alpha'] - merged_sw['alpha_low']) / merged_sw['alpha']
        # Относительные ошибки определения пиковой энергии
        merged_sw['Epeak+'] = (merged_sw['Epeak_hi'] - merged_sw['Epeak']) / merged_sw['Epeak']
        merged_sw['Epeak-'] = (merged_sw['Epeak'] - merged_sw['Epeak_low']) / merged_sw['Epeak']

        # Поправка за единицы потока в каталоге
        merged_sw['S_obs'] = merged_sw['S_obs'] * 1e-7
        merged_sw['S_obs_err'] = merged_sw['S_obs_err'] * 1e-7
        merged_sw['S+-'] = merged_sw['S_obs_err']/merged_sw['S_obs']

        # Заменяем пропущенные относительные ошибки медианными
        data_result = self.replace_missing_values(merged_sw, 'median', ['Epeak+', 'Epeak-', 'S+-', 'alpha+', 'alpha-'])

        # Нижние и верхние абсолютные ошибки пиковой энергии в нашей СО и альфы
        data_result['DEpeak_hi'] = data_result['Epeak'] * data_result['Epeak+'] / 1.654  # 90% ---> 68%
        data_result['DEpeak_low'] = data_result['Epeak'] * data_result['Epeak-'] / 1.654 # 90% ---> 68%
        data_result['Dalpha_hi'] = data_result['alpha'] * data_result['alpha+'] / 1.654  # 90% ---> 68%
        data_result['Dalpha_low'] = data_result['alpha'] * data_result['alpha-'] / 1.654 # 90% ---> 68%
        data_result['S_obs_err'] = data_result['S_obs'] * data_result['S+-'] / 1.654     # 90% ---> 68%

        data_result = data_result[(data_result['Dalpha_hi'] > 0) & (data_result['Dalpha_low'] > 0) &
                                  (data_result['DEpeak_hi'] > 0) & (data_result['DEpeak_low'] > 0)]

        # Убираем строки с NaN
        data_result = data_result.dropna(subset=['S_obs', 'redshift', 'alpha', 'Epeak', 'T90'])
        # Добавляем условия, если есть
        data_result = data_result[
            (data_result['T90'] > self.grb_t90min) & (data_result['T90'] <= self.grb_t90max) &
            (data_result['redshift'] > self.grb_z_min) & (data_result['redshift'] <= self.grb_z_max) &
            (data_result['Epeak'] > self.grb_ep_min) & (data_result['Epeak'] <= self.grb_ep_max) &
            (data_result['Epeak+'] < self.grb_ep_err_max) & (data_result['Epeak-'] < self.grb_ep_err_max)]

        data_result = data_result.reset_index(drop=True)
        return data_result

    def _get_pantheon(self):
        data = pd.read_csv(self.pantheon_path, delimiter='\s+', skiprows=1, usecols=[2, 4, 5],
                           names=['mu', 'dmu', 'zcmb'])
        data_result = data[(data['zcmb'] > self.sn_z_min) & (data['zcmb'] <= self.sn_z_max)]
        return data_result