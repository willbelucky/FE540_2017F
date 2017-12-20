# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 12. 21.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def pca(input_dataframe, alpha):
    # input_dataframe should composed of (indices,) labels, factor 1,2,3,... as columns
    units = input_dataframe.drop(['label'], axis=1)
    units = pd.DataFrame(MinMaxScaler().fit_transform(units))
    n_components = len(units.columns)
    pca = PCA(n_components=n_components)
    pca.fit(units)
    lambdas = []
    for i in range(n_components):
        lambdas.append(pca.explained_variance_ratio_[i])
        if sum(lambdas) < 1 - alpha:
            continue
        else:
            break
    k = len(lambdas)
    eff_pca = []
    for j in range(k):
        eff_pca.append(pca.components_[j])
    eigen_vectors = np.matmul(np.array(input_dataframe)[:, 1:], np.array(eff_pca).T)
    labels = np.array(input_dataframe['label']).reshape(len(np.array(input_dataframe['label'])), 1)
    eigen_vectors = np.hstack((labels, eigen_vectors))

    columns = ['label']
    for i in range(k):
        columns.append('pca' + str(i + 1))

    return pd.DataFrame(eigen_vectors, columns=columns, index=input_dataframe.index)
