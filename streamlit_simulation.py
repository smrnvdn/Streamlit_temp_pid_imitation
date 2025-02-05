import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Добавляем настройку широкого layout'а в начало файла
st.set_page_config(layout="wide")

# Константы для нагрева и охлаждения
TEMP_PARAMS = {
    'heat': {
        't_out': -5.8,  # Целевая температура нагрева
        'k': 0.3,       # Коэффициент нагрева
        'T0': -13       # Начальная температура нагрева
    },
    'cool': {
        't_out': -13,   # Целевая температура охлаждения
        'k': 0.3,       # Коэффициент охлаждения
        'T0': -5.8      # Начальная температура охлаждения
    }
}

# Границы допустимой температуры
TEMP_LIMITS = {
    'min': -9,
    'max': -6
}

def generate_temp_array(params: dict, n_points: int = 100) -> np.ndarray:
    """
    Генерирует массив температур на основе заданных параметров
    """
    return np.array([
        params['t_out'] + np.exp(-params['k'] * i / 2) * (params['T0'] - params['t_out'])
        for i in range(n_points)
    ]).round(2)

# Создаем массивы температур для нагрева и охлаждения
f_lst_heat = generate_temp_array(TEMP_PARAMS['heat'])
f_lst_cool = generate_temp_array(TEMP_PARAMS['cool'])

def control_temperature(current_temp: float = -7, n_iterations: int = 100, temp_min: float = -9, temp_max: float = -6) -> list:
    """
    Функция контроля температуры
    """
    temperature_history = []
    mode = 'heat'  # Начальный режим работы

    for _ in range(n_iterations):
        temperature_history.append(current_temp)

        if mode == 'heat':
            if current_temp >= temp_max:
                mode = 'cool'
                continue

            mask = f_lst_heat > current_temp
            if not np.any(mask):
                break
            current_temp = f_lst_heat[mask][0]

        else:  # mode == 'cool'
            mask = f_lst_cool < current_temp
            if not np.any(mask):
                mode = 'heat'
                continue

            current_temp = f_lst_cool[mask][0]
            if current_temp <= temp_min:
                mode = 'heat'

    return temperature_history

# Заголовок приложения
st.title('Температурный контроль')

# Создаем слайдеры в боковой панели
st.sidebar.header('Параметры системы')

current_temp = st.sidebar.slider(
    'Начальная температура (°C)',
    min_value=-13.0,
    max_value=-5.8,
    value=-7.0,
    step=0.1,
    help='Начальная температура системы'
)

n_iterations = st.sidebar.slider(
    'Количество итераций',
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help='Количество шагов моделирования'
)

temp_min = st.sidebar.slider(
    'Минимальная температура (°C)',
    min_value=-13.0,
    max_value=-6.0,
    value=-9.0,
    step=0.1,
    help='Минимальная допустимая температура'
)

temp_max = st.sidebar.slider(
    'Максимальная температура (°C)',
    min_value=-9.0,
    max_value=-5.8,
    value=-6.0,
    step=0.1,
    help='Максимальная допустимая температура'
)

# Получаем историю температур
temperature_history = control_temperature(current_temp, n_iterations, temp_min, temp_max)

# Создаем график
fig, ax = plt.subplots(figsize=(20, 12))  # Увеличили размер графика еще больше
ax.plot(temperature_history, label='Температура', linewidth=2)  # Увеличили толщину линии
ax.set_title('Температурный контроль', fontsize=24, pad=20)  # Увеличили размер заголовка
ax.set_xlabel('Время (итерации)', fontsize=18, labelpad=10)  # Увеличили размер подписей осей
ax.set_ylabel('Температура (°C)', fontsize=18, labelpad=10)
ax.grid(True, alpha=0.3)  # Сделали сетку менее заметной
ax.axhline(y=temp_max, color='r', linestyle='--', label='Макс. температура', linewidth=2)
ax.axhline(y=temp_min, color='b', linestyle='--', label='Мин. температура', linewidth=2)
ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.15, 1))  # Увеличили размер легенды
ax.tick_params(axis='both', labelsize=14)  # Увеличили размер делений на осях

# Устанавливаем отступы для графика
plt.tight_layout()

# Создаем одну широкую колонку для графика
col1 = st.columns([1])
with col1[0]:
    st.pyplot(fig, use_container_width=True)