"""
Дополнительный функционал для десктопного приложения.
Используется в gui/app.py.
"""
import tkinter as tk


class RangeSlider(tk.Canvas):
    """Двусторонний ползунок для выбора диапазона значений."""
    def __init__(self, parent, min_val, max_val, low_val, high_val,
                 width=200, height=30, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.low_val = float(low_val)
        self.high_val = float(high_val)
        self.width = width
        self.height = height
        self.pad = 15
        self.track_y = height // 2
        self.marker_r = 6

        self._drag_data = {'marker': None}
        self._draw_static()
        self._draw_markers()
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _to_x(self, val):
        frac = (val - self.min_val) / (self.max_val - self.min_val)
        return self.pad + frac * (self.width - 2 * self.pad)

    def _from_x(self, x):
        frac = (x - self.pad) / (self.width - 2 * self.pad)
        return max(self.min_val, min(self.max_val,
                                     self.min_val + frac * (self.max_val - self.min_val)))

    def _draw_static(self):
        self.create_line(self.pad, self.track_y, self.width - self.pad, self.track_y,
                         fill='gray', width=3)

    def _draw_markers(self):
        self.delete('markers')
        x_low = self._to_x(self.low_val)
        x_high = self._to_x(self.high_val)
        self.create_rectangle(x_low, self.track_y - 3, x_high, self.track_y + 3,
                              fill='steelblue', outline='', tags='markers')
        r = self.marker_r
        self.create_oval(x_low - r, self.track_y - r, x_low + r, self.track_y + r,
                         fill='white', outline='black', width=2, tags='markers')
        self.create_oval(x_high - r, self.track_y - r, x_high + r, self.track_y + r,
                         fill='white', outline='black', width=2, tags='markers')

    def _on_click(self, event):
        x, y = event.x, event.y
        dist_low = ((x - self._to_x(self.low_val))**2 + (y - self.track_y)**2)**0.5
        dist_high = ((x - self._to_x(self.high_val))**2 + (y - self.track_y)**2)**0.5
        if dist_low < self.marker_r * 1.5:
            self._drag_data['marker'] = 'low'
        elif dist_high < self.marker_r * 1.5:
            self._drag_data['marker'] = 'high'
        else:
            if abs(x - self._to_x(self.low_val)) < abs(x - self._to_x(self.high_val)):
                self._drag_data['marker'] = 'low'
            else:
                self._drag_data['marker'] = 'high'

    def _on_drag(self, event):
        if self._drag_data['marker'] is None:
            return
        val = self._from_x(event.x)
        if self._drag_data['marker'] == 'low':
            if val >= self.high_val:
                return
            self.low_val = val
        else:
            if val <= self.low_val:
                return
            self.high_val = val
        self._draw_markers()

    def _on_release(self, event):
        self._drag_data['marker'] = None

    def get(self):
        return self.low_val, self.high_val

    def set(self, low, high):
        self.low_val = max(self.min_val, min(self.max_val, float(low)))
        self.high_val = max(self.min_val, min(self.max_val, float(high)))
        if self.low_val >= self.high_val:
            self.high_val = self.low_val + 1e-6
        self._draw_markers()


class SingleSlider(tk.Canvas):
    """Односторонний ползунок (для ep_err_max)."""
    def __init__(self, parent, min_val, max_val, value, width=200, height=30, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.value = float(value)
        self.width = width
        self.height = height
        self.pad = 15
        self.track_y = height // 2
        self.marker_r = 6

        self._drag = False
        self._draw()
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _to_x(self, val):
        frac = (val - self.min_val) / (self.max_val - self.min_val)
        return self.pad + frac * (self.width - 2 * self.pad)

    def _from_x(self, x):
        frac = (x - self.pad) / (self.width - 2 * self.pad)
        return max(self.min_val, min(self.max_val,
                                     self.min_val + frac * (self.max_val - self.min_val)))

    def _draw(self):
        self.delete('all')
        self.create_line(self.pad, self.track_y, self.width - self.pad, self.track_y,
                         fill='gray', width=3)
        x = self._to_x(self.value)
        self.create_line(self.pad, self.track_y, x, self.track_y,
                         fill='steelblue', width=5)
        r = self.marker_r
        self.create_oval(x - r, self.track_y - r, x + r, self.track_y + r,
                         fill='white', outline='black', width=2)

    def _on_click(self, event):
        self._drag = True
        self._update_value(event.x)

    def _on_drag(self, event):
        if self._drag:
            self._update_value(event.x)

    def _on_release(self, event):
        self._drag = False

    def _update_value(self, x):
        self.value = self._from_x(x)
        self._draw()

    def get(self):
        return self.value

    def set(self, val):
        self.value = max(self.min_val, min(self.max_val, float(val)))
        self._draw()