"""
Централизованное управление путями проекта.
Все пути абсолютные, отталкиваются от корня проекта (папка, содержащая core/).
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from pathlib import Path
import sys
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class ProjectPaths:
    """
    Предоставляет абсолютные пути к основным директориям и файлам данных.

    Methods
    -------
    data_dir() -> Path
        Путь к папке data.
    results_dir() -> Path
        Путь к папке results.
    gui_dir() -> Path
        Путь к папке gui.
    swift1_path() -> str
        Путь к SWIFT1.txt.
    swift2_path() -> str
        Путь к SWIFT2.txt.
    pantheon_path() -> str
        Путь к Pantheon.dat.
    """

    def __init__(self):
        # При сборке EXE будет использоваться sys._MEIPASS,
        # но для обычного запуска оставляем относительный корень.
        self.root = Path(__file__).resolve().parent.parent

    def data_dir(self) -> Path:
        """Вернуть путь к папке data."""
        return self.root / 'data'

    def results_dir(self) -> Path:
        """Вернуть путь к папке results."""
        return self.root / 'results'

    def gui_dir(self) -> Path:
        """Вернуть путь к папке gui."""
        return self.root / 'gui'

    # Пути к конкретным файлам данных
    def swift1_path(self) -> str:
        """Вернуть абсолютный путь к SWIFT1.txt."""
        return str(self.data_dir() / 'SWIFT1.txt')

    def swift2_path(self) -> str:
        """Вернуть абсолютный путь к SWIFT2.txt."""
        return str(self.data_dir() / 'SWIFT2.txt')

    def pantheon_path(self) -> str:
        """Вернуть абсолютный путь к Pantheon.dat."""
        return str(self.data_dir() / 'Pantheon.dat')