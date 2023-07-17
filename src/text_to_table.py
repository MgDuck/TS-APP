import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD


def from_text_to_table(data, col_text_name = 'text',\
                        save_mode = 'pickle',
                        train_test_splitting = True,
                        col_sample_name = 'data_sample',\
                        train_sample_name = 'train',\
                        test_sample_name = 'test',\
                        truncate_power = 25,\
                        hf_model = 'flax-sentence-embeddings/stackoverflow_mpnet-base'):
    embed = []
    model = SentenceTransformer(hf_model)

    if train_test_splitting == False:
        data[col_sample_name] = train_sample_name

    for i, text in enumerate(data[data[col_sample_name] == train_sample_name][col_text_name]):
        text_embbedding = model.encode(text)
        embed.append(text_embbedding)

    SVD = TruncatedSVD(n_components = truncate_power)
    SVD.fit(embed)

    svd_train = SVD.transform(embed)
    
    
    res_train_svd = pd.DataFrame()
    res_train_svd['id'] = data[data[col_sample_name] == train_sample_name]['id']
    for i in range(len(svd_train[0])):
        res_train_svd["text_f-"+str(i)] = np.array(svd_train)[:, i]
    if save_mode == 'pickle':
            res_train_svd.to_pickle("outputs/train_tabled_text.pickle")
    else:
        res_train_svd.to_csv("outputs/train_tabled_text.csv")

    if train_test_splitting:
        embed_test = []
        for i, text in enumerate(data[data[col_sample_name] == test_sample_name][col_text_name]):
            text_embbedding = model.encode(text)
            embed_test.append(text_embbedding)
        svd_test = SVD.transform(embed_test)
        res_test_svd = pd.DataFrame()
        res_test_svd['id'] = data[data[col_sample_name] == test_sample_name]['id']
        for i in range(len(svd_test[0])):
            res_test_svd["text_f-"+str(i)] = np.array(svd_train)[:, i]
        if save_mode == 'pickle':
            res_test_svd.to_pickle("outputs/test_tabled_text.pickle")
        else:
            res_test_svd.to_csv("outputs/test_tabled_text.csv")
