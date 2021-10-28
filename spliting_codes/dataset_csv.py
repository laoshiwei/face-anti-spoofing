# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:29:40 2021

@author: Admin
"""

import pandas as pd
import os

df_path = 'C:/Users/Admin/Desktop/CDCN-Face-Anti-Spoofing.pytorch-master/data/nuaa/full_csv.csv'
df = pd.read_csv('C:/Users/Admin/Desktop/face-spoofing/replayattack_extracted/replayattack.csv')
df = df[["Filepath", "Label"]]
for i, row in df.iterrows():
    base = os.path.basename(row['Filepath'])
    df.at[i,'Filepath'] = "images/" + base
    if row['Label'] == "attack":
        df.at[i,'Label'] = 0
    else:
        df.at[i,'Label'] = 1
#df = df.iloc[1: , :]
df.to_csv(df_path, index=False)


print(df)