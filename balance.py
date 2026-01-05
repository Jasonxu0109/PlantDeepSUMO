# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:58:55 2025

@author: jasonxu
"""

import os
import numpy as np
import pandas as pd
os.chdir("C:\\Users\\jasonxu\\Desktop\\PlantDeepSUMO\\CNN")
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import NearMiss
from sklearn.neighbors import NearestNeighbors

for f in [16,18,20,22,24,26,28,30,32,34]:
    print(f)
    for s in [0.4,0.5,0.6,0.7]:
        print(s)
        for data in ['train','test']:
            f1 = "f_"+str(f)+"_c"+str(s)+"_"+data+"_positive_negative.csv"
            df = pd.read_csv(f1)
            encoder = OneHotEncoder()
            
            # 将氨基酸序列转换为数值特征
            sequences = df['Sequence'].apply(lambda x: list(x)).tolist()
            encoded_sequences = encoder.fit_transform(sequences).toarray()
            
            df_encoded = pd.DataFrame(encoded_sequences)
            df_encoded['Label'] = df['Label']
            df_encoded.index = df['Gene']
            df_encoded.head()
            # 将数据分为特征和标签
            X = df_encoded.drop('Label', axis=1)
            y = df_encoded['Label']
            
            # 创建NearMiss对象
            nm = NearMiss()
            
            # 进行降采样
            X_resampled, y_resampled = nm.fit_resample(X, y)
            
            # 创建新的平衡数据框
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled['Label'] = y_resampled
            
            # 查看平衡后的数据
            print(df_resampled['Label'].value_counts())
            
            
            final = df.iloc[nm.sample_indices_]
            
            f2 = "f_"+str(f)+"_c"+str(s)+"_"+data+"_positive_negative_balance.csv"
            final.to_csv(f2,index=False)






