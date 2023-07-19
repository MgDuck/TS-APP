# импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import streamlit as st

import text_to_table
import text_to_translate

from etna.datasets import TSDataset

import time
import pickle
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split

from catboost import Pool, cv
from catboost import CatBoostClassifier


st.set_option("deprecation.showPyplotGlobalUse", False) # отключение предупреждения

st.title("📈TS-APP: предсказание тренда акций") # название приложения
st.header("💼Выбор акции и временного диапазона⌛")




with st.form(key="form"):
    dates = st.slider(
        "Выберите временной диапазон:",
        value=(datetime(2018, 1,1), datetime(2021, 1,1)))
    ticker = st.text_input(
        "Напишите название тикера",
        label_visibility="visible",
        disabled=False,
        placeholder='Например: AAPL'
    )
    submit_button = st.form_submit_button(label="Submit choice")




if submit_button:
    try:
        time.sleep(5)
        prices = yf.download(tickers = ticker,start=dates[0].strftime("%Y-%m-%d"), end=dates[1].strftime("%Y-%m-%d"),\
                                    interval = "1d", prepost = False, repair = True)
        if len(prices.reset_index()) == 0:
            #st.markdown('**:red[Введите верное название тикера]**')
            st.stop()
        prices.to_pickle('prices.pickle')
    except:
        st.warning('Введите верное название тикера', icon="⚠️")
else:
    with open('prices.pickle', 'rb') as f:
        prices = pickle.load(f)



st.write(prices.head())

option = st.selectbox(
        "Выберите целевую переменную",
        ("Open","High","Low","Close","Adj Close","Volume"),
        label_visibility="visible",
        disabled=False
    )

# подготавливаем почву для etna
df_ts = pd.DataFrame()
df_ts['timestamp'] = list(prices.reset_index()['Date'].apply(lambda x: str(x)[:10]))
df_ts['segment'] = ticker
df_ts['target'] = list(prices.reset_index()[option])

################Блок простой визуализации################################


df = TSDataset.to_dataset(df_ts)
ts = TSDataset(df, freq="D")
st.header("📊График ряда")
st.pyplot(ts.plot())

###############Блок преобразования факторов########################


df = prices.reset_index()

# create a moving average

df = df[['Date',option]]

df['target_1'] = list(map(lambda x: int(x>=0), np.array(df[option][1:]) - np.array(df[option][:-1]))) + [-1]
df['target_2'] = list(map(lambda x: int(x>=0), np.array(df[option][2:]) - np.array(df[option][:-2]))) + [-1] * 2
df['target_3'] = list(map(lambda x: int(x>=0), np.array(df[option][3:]) - np.array(df[option][:-3]))) + [-1] * 3
df['target_4'] = list(map(lambda x: int(x>=0), np.array(df[option][4:]) - np.array(df[option][:-4]))) + [-1] * 4
df['target_5'] = list(map(lambda x: int(x>=0), np.array(df[option][5:]) - np.array(df[option][:-5]))) + [-1] * 5
df['target_6'] = list(map(lambda x: int(x>=0), np.array(df[option][6:]) - np.array(df[option][:-6]))) + [-1] * 6
df['target_7'] = list(map(lambda x: int(x>=0), np.array(df[option][7:]) - np.array(df[option][:-7]))) + [-1] * 7
df['target_8'] = list(map(lambda x: int(x>=0), np.array(df[option][8:]) - np.array(df[option][:-8]))) + [-1] * 8
df['target_9'] = list(map(lambda x: int(x>=0), np.array(df[option][9:]) - np.array(df[option][:-9]))) + [-1] * 9
df['target_10'] = list(map(lambda x: int(x>=0), np.array(df[option][10:]) - np.array(df[option][:-10]))) + [-1] * 10

df = df.set_index('Date', inplace=False)


df['ma_5'] = df[option].rolling(window=5).mean()
df['ma_10'] = df[option].rolling(window=10).mean()
df['ma_20'] = df[option].rolling(window=20).mean()
df['ma_40'] = df[option].rolling(window=40).mean()

df['me_5'] = df[option].rolling(window=5).median()
df['me_10'] = df[option].rolling(window=10).median()
df['me_20'] = df[option].rolling(window=20).median()
df['me_40'] = df[option].rolling(window=40).median()

df['max_5'] = df[option].rolling(window=5).max()
df['max_10'] = df[option].rolling(window=10).max()
df['max_20'] = df[option].rolling(window=20).max()
df['max_40'] = df[option].rolling(window=40).max()

df['std_5'] = df[option].rolling(window=5).std()
df['std_10'] = df[option].rolling(window=10).std()
df['std_20'] = df[option].rolling(window=20).std()
df['std_40'] = df[option].rolling(window=40).std()

df['price_diff_1'] = df[option].diff()
df['price_diff_2'] = df[option].diff(2)
df['price_diff_3'] = df[option].diff(3)
df['price_diff_4'] = df[option].diff(4)
df['price_diff_5'] = df[option].diff(5)
df['price_diff_6'] = df[option].diff(6)


df['price_div_1'] = df[option] / df[option].shift(1)
df['price_div_2'] = df[option] / df[option].shift(2)
df['price_div_3'] = df[option] / df[option].shift(3)
df['price_div_4'] = df[option] / df[option].shift(4)
df['price_div_5'] = df[option] / df[option].shift(5)
df['price_div_6'] = df[option] / df[option].shift(6)


df['lag_1'] = df[option].shift(1)
df['lag_2'] = df[option].shift(2)
df['lag_3'] = df[option].shift(3)
df['lag_4'] = df[option].shift(4)
df['lag_5'] = df[option].shift(5)
df['lag_6'] = df[option].shift(6)
df['lag_7'] = df[option].shift(7)
df['lag_8'] = df[option].shift(8)

df = df.dropna(axis=0)
df = df[df['target_10'] >= 0]

need_columns = ['target_1','target_2','target_3','target_4','target_5','target_6','target_7','target_8','target_9','target_10'] # столбцы нужные для удаления 


st.header("🏁Обучение моделей CatBoost")

run_model = st.radio(
    "Сделать кросс-валидацию?",
    ("Нет", "Да"))

if run_model == "Нет":
    st.stop()
else:
    pass


for i in range(1,11):
        train = df
        X = train.drop(need_columns, axis=1).fillna(0) 
        y = train['target_{}'.format(i)]
        params = {"iterations": 300,
                    "depth": 4,
                    "loss_function": "Logloss",
                    'eval_metric': 'Accuracy',
                    'learning_rate': 0.03,
                    'custom_metric' : ['MCC','Precision','Recall','F1','Logloss','NormalizedGini'],
                    "verbose": False,
                    "use_best_model": True}

        cv_dataset = Pool(data=X,
                            label=y, cat_features=[])

        scores_base = cv(cv_dataset,
                    params,
                    return_models=True,
                    fold_count=6,
                    shuffle=False,
                    plot=False,
                    type='TimeSeries')
        

        bests = {}
        bests['MCC'] = []
        bests['F1'] = []
        bests['Accuracy'] = []
        bests['NormalizedGini'] = []
        for j in range(0,6):
            res = pd.read_csv('catboost_info/fold-{}/test_error.tsv'.format(j), sep='\t')
            bests['MCC'] += [res['MCC'].max()]
            bests['F1'] += [res['F1'].max()]
            bests['Accuracy'] += [res['Accuracy'].max()]
            bests['NormalizedGini'] += [res['NormalizedGini'].max()]

        st.write('Лучшие результаты метрик для предсказания тренда {}-го дня на каждом фолде'.format(i))
        st.write(pd.DataFrame(bests))
