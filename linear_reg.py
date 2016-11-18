import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_y = train_df['loss']
test_id = test_df['id']

train_df.drop('loss',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)
train_df.drop('id',axis=1,inplace=True)

to_drop = []

train_df.drop(to_drop,axis=1,inplace=True)
test_df.drop(to_drop,axis=1,inplace=True)

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

train_X = train_df.as_matrix()
train_Y = train_y.as_matrix()
test_X = test_df.as_matrix()

print train_X.shape
print test_X.shape
print train_Y.shape

clf = linear_model.LinearRegression()
scores = cross_val_score(clf,train_X, train_Y)
print scores

to_submit = False
if to_submit:
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(train_X,train_Y)
    pred = lin_reg.predict(test_X)

    _submit = open('submit.csv','w')
    _submit.write('id,loss\n')
    for i in range(0,len(pred)):
        _submit.write(str(test_id[i])+","+str(pred[i])+"\n")

    _submit.flush()
    _submit.close()