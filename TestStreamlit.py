# import module
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# temsorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Activation, Dropout, LSTM
import tensorflow as tf

def Get_y():
    # Загружаем и инициализируем
    minmax_scaler_y_ns = MinMaxScaler()
    minmax_scaler_X_ns = MinMaxScaler()
    minmax_scaler_y_ns = joblib.load('y_ns.mms')
    minmax_scaler_X_ns = joblib.load('X_ns.mms')
    model_ns = tf.keras.models.load_model('model_ns.h5')
    X_temp = np.zeros(shape=(1, 12))
    X_temp[0, 0] = level1
    X_temp[0, 1] = level2
    X_temp[0, 2] = level3
    X_temp[0, 3] = level4
    X_temp[0, 4] = level5
    X_temp[0, 5] = level6
    X_temp[0, 6] = level7
    X_temp[0, 7] = level8
    X_temp[0, 8] = level9
    X_temp[0, 9] = level10
    X_temp[0, 10] = level11
    X_temp[0, 11] = level12
    
    # Нормализуем вход
    X_ns = minmax_scaler_X_ns.transform(X_temp)
    # Получем прогноз
    y_ns = model_ns.predict(X_ns)
    # Обратно бреобразовываем целевую переменную
    y_ns = minmax_scaler_y_ns.inverse_transform(y_ns)
    
    return y_ns
    
# Title
st.header('Приложение для прогноза матрица-наполнитель')
st.subheader('Дикое, но симпатичное!')

# slider
level1 = st.slider('Плотность, кг/м3', 1731, 2208)
level2 = st.slider('Модуль упругости, ГПа', 2, 1911)
level3 = st.slider('Количество отвердителя, м%', 17.74, 138.95)
level4 = st.slider('Содержание эпоксидных групп, %_2', 14.255, 33.000)
level5 = st.slider('Температура вспышки, с_2', 100.00, 413.27)
level6 = st.slider('Поверхностная плотность, г/м2', 1, 1399)
level7 = st.slider('Потребление смолы, г/м2', 33.80, 414.59)
level8 = st.slider('Шаг нашивки', 0.00, 14.44)
level9 = st.slider('Плотность нашивки', 0.0, 104.0)
level10 = st.radio('Угол нашики 0 град', (0, 90))
if level10 == 90:
    level10 = 1
level11 = st.slider('Модуль упругости при растяжении, ГПа', 64.05, 82.68)
level12 = st.slider('Прочность при растяжении, МПа', 1036, 3849)

# Create a button, that when clicked, shows a result
if (st.button( 'Пересчитать' )):
    st.success(f'Соотношение матрица-наполнитель предлагается: {Get_y()}')
