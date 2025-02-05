import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

# Добавляем настройку широкого layout'а в начало файла
st.set_page_config(layout="wide")

# Общие параметры для расчета температуры
TEMP_PARAMS = {
    't_out': -5.8,  # Целевая температура
    'k': 0.3,       # Коэффициент нагрева/охлаждения
    'T0': -13.0     # Начальная температура
}

# Границы допустимой температуры
TEMP_LIMITS = {
    'min': -9,
    'max': -6
}

def generate_temp_array(params: dict, is_heating: bool = True, n_points: int = 100) -> np.ndarray:
    """
    Генерирует массив температур на основе заданных параметров
    Args:
        params: словарь с параметрами
        is_heating: True для нагрева, False для охлаждения
        n_points: количество точек
    """
    t_out = params['t_out'] if is_heating else params['T0']
    t0 = params['T0'] if is_heating else params['t_out']

    return np.array([
        t_out + np.exp(-params['k'] * i / 2) * (t0 - t_out)
        for i in range(n_points)
    ]).round(2)

# Создаем массивы температур для нагрева и охлаждения
f_lst_heat = generate_temp_array(TEMP_PARAMS, is_heating=True)
f_lst_cool = generate_temp_array(TEMP_PARAMS, is_heating=False)

def control_temperature(current_temp: float = -7, n_iterations: int = 100, temp_min: float = -9, temp_max: float = -6) -> list:
    """
    Функция контроля температуры
    """
    temperature_history = []
    mode = 'heat'  # Режим нагрева

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

# Две колонки: одна для параметров, другая для графика
col_params, col_graph = st.columns([0.25, 0.75])
with col_params:
    st.markdown('##### Параметры системы')


    # Параметры моделирования
    temp_params = {
        't_out': st.number_input(
            'Целевая температура (°C)',
            min_value=-15.0,
            max_value=0.0,
            value=float(TEMP_PARAMS['t_out']),
            step=0.1,
            help='Конечная температура нагрева/охлаждения',
            key='t_out_input'
        ),
        'k': st.number_input(
            'Коэффициент нагрева/охлаждения',
            min_value=0.1,
            max_value=1.0,
            value=float(TEMP_PARAMS['k']),
            step=0.1,
            help='Скорость изменения температуры',
            key='k_input'
        ),
        'T0': st.number_input(
            'Начальная температура (°C)',
            min_value=-15.0,
            max_value=0.0,
            value=float(TEMP_PARAMS['T0']),
            step=0.1,
            help='Начальная температура системы',
            key='T0_input'
        )
    }

    st.divider()

    # Параметры симуляции
    n_iterations = st.slider(
        'Количество итераций',
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help='Количество шагов моделирования'
    )

    temp_min = st.slider(
        'Минимальная температура (°C)',
        min_value=-13.0,
        max_value=-6.0,
        value=-9.0,
        step=0.1,
        help='Минимальная допустимая температура'
    )

    temp_max = st.slider(
        'Максимальная температура (°C)',
        min_value=-9.0,
        max_value=-5.8,
        value=-6.0,
        step=0.1,
        help='Максимальная допустимая температура'
    )

    current_temp = st.slider(
        'Текущая температура (°C)',
        min_value=temp_min,
        max_value=temp_max,
        value=min(max(temp_min, -7.0), temp_max),
        step=0.1,
        help='Текущая температура системы'
    )

    st.divider()

# Создаем массивы температур с обновленными параметрами
f_lst_heat = generate_temp_array(temp_params, is_heating=True)
f_lst_cool = generate_temp_array(temp_params, is_heating=False)

# Получаем историю температур
temperature_history = control_temperature(current_temp, n_iterations, temp_min, temp_max)

def to_excel():
    # Создаем DataFrame с данными
    df = pd.DataFrame({
        'Время (итерации)': range(len(temperature_history)),
        'Температура': temperature_history,
        'Максимальная температура': [temp_max] * len(temperature_history),
        'Минимальная температура': [temp_min] * len(temperature_history)
    })

    # Создаем буфер в памяти для Excel файла
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Температурный контроль', index=False)

        # Получаем объект workbook и worksheet
        workbook = writer.book
        worksheet = writer.sheets['Температурный контроль']

        # Добавляем форматирование
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })

        # Применяем форматирование к заголовкам
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 15)  # Устанавливаем ширину столбцов

    return buffer

# Кнопка выгрузки данных
if st.button('📥 Выгрузить данные в Excel'):
    buffer = to_excel()
    st.download_button(
        label='📊 Скачать Excel файл',
        data=buffer.getvalue(),
        file_name='temperature_control.xlsx',
        mime='application/vnd.ms-excel'
    )

with col_graph:
    # Создаем график с помощью Plotly
    fig = go.Figure()

    # Добавляем основную линию температуры
    fig.add_trace(go.Scatter(
        x=list(range(len(temperature_history))),
        y=temperature_history,
        name='Температура',
        line=dict(width=2)
    ))

    # Добавляем линии минимальной и максимальной температуры
    fig.add_trace(go.Scatter(
        x=[0, len(temperature_history)],
        y=[temp_max, temp_max],
        name='Макс. температура',
        line=dict(dash='dash', color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=[0, len(temperature_history)],
        y=[temp_min, temp_min],
        name='Мин. температура',
        line=dict(dash='dash', color='blue', width=2)
    ))

    # Настраиваем внешний вид графика
    fig.update_layout(
        xaxis_title={
            'text': 'Время (итерации)',
            'font': dict(size=18)
        },
        yaxis_title={
            'text': 'Температура (°C)',
            'font': dict(size=18)
        },
        showlegend=True,
        legend=dict(
            font=dict(size=16),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15
        ),
        hovermode='x unified',
        height=800,
        margin=dict(t=20)  # Уменьшаем верхний отступ
    )

    # Отображаем график на всю ширину контейнера
    st.plotly_chart(fig, use_container_width=True)