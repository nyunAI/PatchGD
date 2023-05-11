import pandas as pd
train_csv = pd.read_csv('./train_100_full.csv')
val_csv = pd.read_csv('./val_100_10_1.csv')

print(train_csv.head())
print(val_csv.head())

train_classes = train_csv['class'].unique()
val_classes = val_csv['class'].unique()
train_classes.sort()
val_classes.sort()

print((train_classes == val_classes).all())
