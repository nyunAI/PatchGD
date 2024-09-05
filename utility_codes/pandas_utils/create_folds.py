import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

TRAIN_CSV_PATH = ''
KFOLD_CSV_PATH = './'

df = pd.read_csv(TRAIN_CSV_PATH)

df["kfold"] = -1  
df = df.sample(frac=1).reset_index(drop=True)  
y = df.isup_grade.values  
kf = StratifiedKFold(n_splits=5)  
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):  
    df.loc[v_, 'kfold'] = f  

df.to_csv(KFOLD_CSV_PATH)