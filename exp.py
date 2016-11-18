import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_y = train_df['loss']
test_id = test_df['id']

train_df.drop('loss',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)
train_df.drop('id',axis=1,inplace=True)

features = train_df.columns
cat_feat = []
cont_feat = []

for feature in features:
    if feature.startswith('cat'):
        cat_feat.append(feature)
    else:
        cont_feat.append(feature)

for feature in cat_feat:
    enc = LabelEncoder()
    enc.fit(pd.concat([train_df[feature],test_df[feature]],ignore_index=True))
    train_df[feature] = enc.transform(train_df[feature])
    test_df[feature] = enc.transform(test_df[feature])

print(train_df.corr())