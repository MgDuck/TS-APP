import time
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect_langs

def translating(data, col_text_name = 'text', tr_language = 'en', save_mode = 'pickle'):
    res = []
    n = 0
    for i in range(len(data[col_text_name])):
        det = detect_langs(data[col_text_name][i])
        if det[0].lang == tr_language and det[0].prob > 0.9:
            res.append(data["summary"][i])
            continue

        if (n + 1) % 100 == 0:
            time.sleep(77)
        translated = GoogleTranslator(source='auto', target=tr_language).translate(text=data[col_text_name][i])
        res.append(translated)
        n = n + 1
    data["translated_text"] = res
    if save_mode == 'csv':
        data.to_csv("outputs/data_plus_translated.csv", sep=",", index=False)
    else:
        data.to_pickle("outputs/data_plus_translated.pickle")