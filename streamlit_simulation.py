import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

# Добавляем настройку широкого layout'а в начало файла
st.set_page_config(layout="wide")

# Уменьшаем отступ от заголовка
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Общие параметры для расчета температуры
TEMP_PARAMS = {
    'heat': {
        't_out': -5.8,  # Целевая температура нагрева
        'k': 0.3,       # Коэффициент нагрева
        'T0': -13.0     # Начальная температура для нагрева
    },
    'cool': {
        't_out': -9.0,  # Целевая температура охлаждения
        'k': 0.3,       # Коэффициент охлаждения
        'T0': -5.8       # Начальная температура для охлаждения
    }
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
    mode = 'heat' if is_heating else 'cool'
    params = params[mode]  # Получаем параметры для нужного режима

    t_out = params['t_out']
    t0 = params['T0']

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
    temperature_history = [current_temp]  # Добавляем начальную температуру
    mode = 'heat'  # Режим нагрева

    for _ in range(n_iterations - 1):  # Уменьшаем на 1, так как начальная точка уже добавлена
        if mode == 'heat':
            if current_temp >= temp_max:
                mode = 'cool'
                continue

            mask = f_lst_heat > current_temp
            if not np.any(mask):
                break
            current_temp = f_lst_heat[mask][0]

        else:  # mode == 'cool'
            if current_temp <= temp_min:
                mode = 'heat'
                continue

            mask = f_lst_cool < current_temp
            if not np.any(mask):
                mode = 'heat'
                continue
            current_temp = f_lst_cool[mask][0]

        temperature_history.append(current_temp)  # Добавляем точку после изменения температуры

    return temperature_history

# Заголовок приложения
st.title('Температурный контроль')

def to_excel():
    # Создаем DataFrame с данными температур
    df_temps = pd.DataFrame({
        'Время (итерации)': range(len(temperature_history)),
        'Температура': temperature_history,
        'Максимальная температура': [temp_max] * len(temperature_history),
        'Минимальная температура': [temp_min] * len(temperature_history)
    })

    # Создаем DataFrame с параметрами системы
    df_params = pd.DataFrame({
        'Параметр': [
            'Количество итераций',
            'Минимальная температура (°C)',
            'Максимальная температура (°C)',
            'Начальная температура (°C)',
            'Целевая температура нагрева (°C)',
            'Коэффициент нагрева',
            'Начальная температура нагрева (°C)',
            'Целевая температура охлаждения (°C)',
            'Коэффициент охлаждения',
            'Начальная температура охлаждения (°C)',
        ],
        'Значение': [
            n_iterations,
            temp_min,
            temp_max,
            current_temp,
            temp_params['heat']['t_out'],
            temp_params['heat']['k'],
            temp_params['heat']['T0'],
            temp_params['cool']['t_out'],
            temp_params['cool']['k'],
            temp_params['cool']['T0'],
        ]
    })

    # Создаем буфер в памяти для Excel файла
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Записываем данные на разные листы
        df_temps.to_excel(writer, sheet_name='Температурный контроль', index=False)
        df_params.to_excel(writer, sheet_name='Параметры системы', index=False)

        # Получаем объект workbook и worksheet'ы
        workbook = writer.book
        worksheet_temps = writer.sheets['Температурный контроль']
        worksheet_params = writer.sheets['Параметры системы']

        # Добавляем форматирование
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })

        # Применяем форматирование к заголовкам на листе температур
        for col_num, value in enumerate(df_temps.columns.values):
            worksheet_temps.write(0, col_num, value, header_format)
            worksheet_temps.set_column(col_num, col_num, 15)

        # Применяем форматирование к заголовкам на листе параметров
        for col_num, value in enumerate(df_params.columns.values):
            worksheet_params.write(0, col_num, value, header_format)
            worksheet_params.set_column(col_num, col_num, 30)  # Увеличиваем ширину для читаемости

    return buffer

# Две колонки: одна для параметров, другая для графика
col_params, col_graph = st.columns([0.22, 0.78])
with col_params:
    st.markdown('##### Параметры системы')
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
        min_value=temp_min + 0.1,  # Гарантируем, что temp_max строго больше temp_min
        max_value=-5.8,
        value=-6.0 if -6.0 > temp_min else temp_min + 0.1,
        step=0.1,
        help='Максимальная температура (должна быть больше минимальной)'
    )

    current_temp = st.slider(
        'Текущая температура (°C)',
        min_value=temp_min,
        max_value=temp_max,
        value=min(max(temp_min, -7.0), temp_max),
        step=0.1,
        help='Текущая температура системы'
    )

    # Добавляем выбор ступени производительности
    performance_level = st.radio(
        "Ступень производительности",
        options=["Стандартная", "Ступень 1", "Ступень 2", "Ступень 3", "Ступень 4"],
        index=0,
        help="Выберите ступень производительности системы"
    )


    # Определяем параметры для разных ступеней производительности
    PERFORMANCE_PARAMS = {
        "Стандартная": {
            'heat': {'k': 0.3},
            'cool': {'k': 0.3}
        },
        "Ступень 1": {
            'heat': {'k': 0.3},
            'cool': {'k': 0.03}
        },
        "Ступень 2": {
            'heat': {'k': 0.3},
            'cool': {'k': 0.06}
        },
        "Ступень 3": {
            'heat': {'k': 0.3},
            'cool': {'k': 0.09}
        },
        "Ступень 4": {
            'heat': {'k': 0.3},
            'cool': {'k': 0.12}
        }
    }

    with st.expander("Дополнительные параметры"):
        temp_params = {
            'heat': {
                't_out': st.number_input(
                    'Целевая температура нагрева (°C)',
                    min_value=-15.0,
                    max_value=0.0,
                    value=float(TEMP_PARAMS['heat']['t_out']),
                    step=0.1,
                    help='Конечная температура нагрева',
                    key='t_out_heat_input'
                ),
                'k': st.number_input(
                    'Коэффициент нагрева',
                    min_value=0.01,
                    max_value=1.0,
                    value=PERFORMANCE_PARAMS[performance_level]['heat']['k'],
                    step=0.01,
                    help='Скорость нагрева',
                    key='k_heat_input'
                ),
                'T0': st.number_input(
                    'Начальная температура нагрева (°C)',
                    min_value=-15.0,
                    max_value=0.0,
                    value=float(TEMP_PARAMS['heat']['T0']),
                    step=0.1,
                    help='Начальная температура для нагрева',
                    key='T0_heat_input'
                )
            },
            'cool': {
                't_out': st.number_input(
                    'Целевая температура охлаждения (°C)',
                    min_value=-15.0,
                    max_value=0.0,
                    value=float(TEMP_PARAMS['cool']['t_out']),
                    step=0.1,
                    help='Конечная температура охлаждения',
                    key='t_out_cool_input'
                ),
                'k': st.number_input(
                    'Коэффициент охлаждения',
                    min_value=0.01,
                    max_value=1.0,
                    value=PERFORMANCE_PARAMS[performance_level]['cool']['k'],
                    step=0.01,
                    help='Скорость охлаждения',
                    key='k_cool_input'
                ),
                'T0': st.number_input(
                    'Начальная температура охлаждения (°C)',
                    min_value=-15.0,
                    max_value=0.0,
                    value=float(TEMP_PARAMS['cool']['T0']),
                    step=0.1,
                    help='Начальная температура для охлаждения',
                    key='T0_cool_input'
                )
            }
        }

    st.divider()

# Создаем массивы температур с обновленными параметрами
f_lst_heat = generate_temp_array(temp_params, is_heating=True)
f_lst_cool = generate_temp_array(temp_params, is_heating=False)

# Получаем историю температур
temperature_history = control_temperature(current_temp, n_iterations, temp_min, temp_max)

with col_params:
    # Кнопка выгрузки данных
    if st.button('📥 Выгрузить данные в Excel'):
        st.download_button(
            label='📊 Скачать Excel файл',
            data=to_excel().getvalue(),
            file_name='temperature_control.xlsx',
            mime='application/vnd.ms-excel'
        )


with col_graph:
    # Добавляем слайдер для управления высотой графика
    graph_height = st.slider(
        'Высота графика',
        min_value=400,
        max_value=800,
        value=650,
        step=50,
        help='Измените высоту графика'
    )

    # Создаем график с помощью Plotly
    fig = go.Figure()

    # Определяем цвет линии в зависимости от ступени
    line_color = 'blue' if performance_level == "Стандартная" else 'red'

    # Добавляем основную линию температуры с соответствующим цветом
    fig.add_trace(go.Scatter(
        x=list(range(len(temperature_history))),
        y=temperature_history,
        name=f'Температура ({performance_level})',
        line=dict(width=2, color=line_color)
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
        height=graph_height,  # Используем значение из слайдера
        margin=dict(t=20, b=20, l=20, r=20),
        # Добавляем возможность рисовать аннотации
        newshape=dict(line_color='red'),
        modebar_add=['drawline', 'drawopenpath', 'eraseshape']
    )

    # Отображаем график на всю ширину контейнера
    st.plotly_chart(fig, use_container_width=True)