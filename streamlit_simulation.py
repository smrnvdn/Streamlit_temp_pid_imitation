import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

# Настройка интерфейса
st.set_page_config(layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Базовые параметры системы
PERFORMANCE_PARAMS = {
    "Стандартная": {'heat': {'k': 0.3}, 'cool': {'k': 0.3}},
    "Ступень 1": {'cool': {'k': 0.05}},
    "Ступень 2": {'cool': {'k': 0.1}},
    "Ступень 3": {'cool': {'k': 0.2}},
    "Ступень 4": {'cool': {'k': 0.3}}
}

def generate_temp_array(params: dict, is_heating: bool = True, n_points: int = 1000) -> np.ndarray:
    """Генерирует массив температур на основе экспоненциальной модели"""
    mode = 'heat' if is_heating else 'cool'
    params = params[mode]
    return np.array([
        params['t_out'] + np.exp(-params['k'] * i / 2) * (params['T0'] - params['t_out'])
        for i in range(n_points)
    ]).round(2)

def control_temperature(current_temp: float, n_iterations: int, temp_min: float, temp_max: float) -> list:
    """Стандартный контроль температуры"""
    temperature_history = [current_temp]
    mode = 'heat'

    for _ in range(n_iterations - 1):
        if mode == 'heat':
            if current_temp >= temp_max:
                mode = 'cool'
                continue
            mask = f_lst_heat > current_temp
            if not np.any(mask):
                break
            current_temp = f_lst_heat[mask][0]
        else:
            if current_temp <= temp_min:
                mode = 'heat'
                continue
            mask = f_lst_cool < current_temp
            if not np.any(mask):
                mode = 'heat'
                continue
            current_temp = f_lst_cool[mask][0]
        temperature_history.append(current_temp)

    return temperature_history

def control_temperature_combined(current_temp: float, n_iterations: int, temp_min: float, temp_max: float,
                               pid_setpoint: float, Kp: float, Ki: float, TEMP_PARAMS: dict, dt: float) -> tuple:
    """Комбинированный контроль температуры с ПИД-регулятором"""
    temperature_history = [current_temp]
    mode = "heat"
    steps_lst = []
    errors_lst = []
    sum_of_steps = 0
    dt_counter = []

    for _ in range(n_iterations - 1):
        if mode == "heat":
            if current_temp >= temp_max:
                mode = "cool"
                dt_counter = []
            else:
                mask = f_lst_heat > current_temp
                if not np.any(mask):
                    break
                current_temp = f_lst_heat[mask][0]
                temperature_history.append(current_temp)
                steps_lst.append(temp_min)
                errors_lst.append(0)
                continue

        if mode == "cool":
            if current_temp <= temp_min:
                mode = "heat"
                continue

            error = max(current_temp - pid_setpoint, 0)
            dt_counter.append(error)
            pid_output = Kp * error + Ki * sum(dt_counter)

            if len(dt_counter) > dt:
                dt_counter.pop(0)

            # Определение ступени охлаждения
            if pid_output > 0.6:
                num_steps, performance_level = 4, 'Ступень 4'
            elif pid_output > 0.3:
                num_steps, performance_level = 3, 'Ступень 3'
            elif pid_output > 0.15:
                num_steps, performance_level = 2, 'Ступень 2'
            else:
                num_steps, performance_level = 1, 'Ступень 1'

            errors_lst.append(pid_output)

            # Обновление параметров охлаждения
            temp_params_pid = {
                'cool': {
                    't_out': float(TEMP_PARAMS['cool']['t_out']),
                    'k': float(PERFORMANCE_PARAMS[performance_level]['cool']['k']),
                    'T0': float(TEMP_PARAMS['cool']['T0']),
                }
            }
            f_lst_cool = generate_temp_array(temp_params_pid, is_heating=False)

            mask = f_lst_cool < current_temp
            if not np.any(mask):
                mode = 'heat'
                continue
            current_temp = f_lst_cool[mask][0]

            temperature_history.append(current_temp)
            steps_lst.append(num_steps + temp_min)
            sum_of_steps += num_steps

    return temperature_history, steps_lst, errors_lst, sum_of_steps

def create_plot(temperature_history, temperature_history_pid, num_steps_lst, errors_lst, steps_for_sum,
                temp_min, temp_max, graph_height):
    """Создание графика"""
    fig = go.Figure()

    # Основные графики
    fig.add_trace(go.Scatter(
        x=list(range(len(temperature_history))),
        y=temperature_history,
        name='Температура',
        line=dict(width=2, color='blue'),
        visible='legendonly'
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(temperature_history_pid))),
        y=temperature_history_pid,
        name='ПИД-регулятор охлаждения',
        line=dict(width=3, dash='dot', color='green')
    ))

    # Ступени и ошибки ПИД
    steps_text = []
    if num_steps_lst:
        steps_text.append(f"{int(num_steps_lst[0] - temp_min)}")
        for i in range(1, len(num_steps_lst)):
            steps_text.append(f"{int(num_steps_lst[i] - temp_min)}" if num_steps_lst[i] != num_steps_lst[i-1] else "")

    fig.add_trace(go.Scatter(
        x=list(range(len(num_steps_lst))),
        y=num_steps_lst,
        name='Ступени ПИД-регулятора',
        line=dict(width=3, dash='dot', color='orange'),
        mode="lines+markers+text",
        text=steps_text,
        textposition="top center",
        textfont=dict(size=15, color="black")
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(errors_lst))),
        y=errors_lst,
        name='Ошибки ПИД-регулятора',
        line=dict(width=3, dash='dot', color='red'),
        visible="legendonly"
    ))

    # Границы температуры
    for temp, name, color in [(temp_max, 'Макс.', 'red'), (temp_min, 'Мин.', 'blue')]:
        fig.add_trace(go.Scatter(
            x=[0, len(temperature_history)],
            y=[temp, temp],
            name=f'{name} температура',
            line=dict(dash='dash', color=color, width=2)
        ))

    # Настройка внешнего вида
    fig.update_layout(
        xaxis_title={'text': "Время (итерации)", 'font': dict(size=18)},
        yaxis_title={'text': "Температура (°C)", 'font': dict(size=18)},
        showlegend=True,
        legend=dict(font=dict(size=16), yanchor="top", y=0.99, xanchor="right", x=1.15),
        hovermode="x unified",
        height=graph_height,
        margin=dict(t=20, b=20, l=20, r=20),
        newshape=dict(line_color="red"),
        modebar_add=["drawline", "drawopenpath", "eraseshape"]
    )

    fig.add_annotation(
        x=0.5, y=1.03,
        xref="paper", yref="paper",
        text=f"Суммарное количество ступеней: {steps_for_sum}",
        showarrow=False,
        font=dict(size=16, color="black")
    )

    return fig

# Основной интерфейс
st.title('Температурный контроль')
col_params, col_graph = st.columns([0.22, 0.78])

# Панель параметров
with col_params:
    st.markdown('##### Параметры системы')

    n_iterations = st.slider('Количество итераций', 10, 500, 100, 10)
    setpoint_temp = st.slider('Целевая температура (°C)', -13.0, -6.0, -7.5, 0.1)
    deadband_temp = st.slider('Зона нечувствительности (°C)', 0.0, 10.0, 7.0, 1.0)

    temp_min = setpoint_temp - deadband_temp / 10
    temp_max = setpoint_temp + deadband_temp / 10
    current_temp = st.slider('Текущая температура (°C)',
                           temp_min, temp_max-0.1,
                           min(max(temp_min, -7.0), temp_max), 0.1)

    with st.expander("Параметры ПИД-регулятора"):
        Kp_input = st.number_input("Kp", value=1.0, step=0.1, format="%.3f")
        Ki_input = st.number_input("Ki", value=1.0, step=0.01, format="%.4f")
        dt_input = st.number_input("dt", value=1, step=1, min_value=1)
        # Kd_input = st.number_input("Kd", value=0.0, step=0.01, format="%.3f")

    # Параметры температуры
    TEMP_PARAMS = {
        'heat': {'t_out': -5.8, 'k': 0.3, 'T0': -13.0},
        'cool': {'t_out': float(temp_min), 'k': float(PERFORMANCE_PARAMS["Стандартная"]['cool']['k']), 'T0': float(temp_max)}
    }

    with st.expander("Дополнительные параметры"):
        temp_params = {
            'heat': {
                't_out': st.number_input('Целевая температура нагрева (°C)', -15.0, 0.0, float(TEMP_PARAMS['heat']['t_out']), 0.1),
                'k': st.number_input('Коэффициент нагрева', 0.01, 1.0, PERFORMANCE_PARAMS["Стандартная"]['heat']['k'], 0.01),
                'T0': st.number_input('Начальная температура нагрева (°C)', -15.0, 0.0, float(temp_min), 0.1)
            }
        }

# Расчеты
f_lst_heat = generate_temp_array(TEMP_PARAMS, is_heating=True)
f_lst_cool = generate_temp_array(TEMP_PARAMS, is_heating=False)

temperature_history = control_temperature(current_temp, n_iterations, temp_min, temp_max)

TEMP_PARAMS_PID = {'cool': {'t_out': temp_min, 'T0': temp_max}}
temperature_history_pid, num_steps_lst, errors_lst, steps_for_sum = control_temperature_combined(
    current_temp, n_iterations, temp_min, temp_max,
    setpoint_temp, Kp_input, Ki_input, TEMP_PARAMS_PID, dt_input
)

# Отображение графика
with col_graph:
    graph_height = st.slider('Высота графика', 400, 800, 650, 50)
    fig = create_plot(temperature_history, temperature_history_pid, num_steps_lst, errors_lst,
                     steps_for_sum, temp_min, temp_max, graph_height)
    st.plotly_chart(fig, use_container_width=True)
