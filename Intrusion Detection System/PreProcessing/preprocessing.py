import numpy as np
 import pandas as pd
 from sklearn.preprocessing import MinMaxScaler
 from sklearn import preprocessing
 train = pd.read_csv("UNSW_NB15_training-set.csv")
 test = pd.read_csv("UNSW_NB15_testing-set.csv")
 data = pd.concat([train, test], axis=0)
 data[data['service']=='-']
 data['service'].replace('-',np.nan,inplace=True)
 data.dropna(inplace=True)
 features = pd.read_csv("NUSW-NB15_features.csv",encoding='cp1252')
 features['Type '] = features['Type '].str.lower()
 nominal_names = features['Name'][features['Type ']=='nominal']
 integer_names = features['Name'][features['Type ']=='integer']
 binary_names = features['Name'][features['Type ']=='binary']
 float_names = features['Name'][features['Type ']=='float']

 cols = data.columns
 nominal_names = cols.intersection(nominal_names)
 integer_names = cols.intersection(integer_names)
 binary_names = cols.intersection(binary_names)
 float_names = cols.intersection(float_names)

 for c in integer_names:
 pd.to_numeric(data[c])
 for c in binary_names
pd.to_numeric(data[c])
 for c in float_names:
 pd.to_numeric(data[c])

 num_col = data.select_dtypes(include='number').columns
 cat_col = data.columns.difference(num_col)
 cat_col = cat_col[1:]
 data_cat = data[cat_col].copy()
 data_cat = pd.get_dummies(data_cat,columns=cat_col)
 data = pd.concat([data, data_cat],axis=1)
 data.drop(columns=cat_col,inplace=True)
 num_col = list(data.select_dtypes(include='number').columns)
 num_col.remove('id')
 num_col.remove('label')
 print(num_col)

 minmax_scale = MinMaxScaler(feature_range=(0, 1))
 def normalization(df,col):
 for i in col:
 arr = df[i]
 arr = np.array(arr)
 df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
 return df
 data = normalization(data.copy(),num_col)

 bin_label = pd.DataFrame(data.label.map(
 lambda x:'normal' if x==0 else 'abnormal')
 )

 bin_data = data.copy()
 bin_data['label'] = bin_label
le1 = preprocessing.LabelEncoder()
 enc_label = bin_label.apply(le1.fit_transform)
 bin_data['label'] = enc_label
 multi_data = data.copy()
 multi_label = pd.DataFrame(multi_data.attack_cat)
 multi_data = pd.get_dummies(multi_data,columns=['attack_cat'])

 le2 = preprocessing.LabelEncoder()
 enc_label = multi_label.apply(le2.fit_transform)
 multi_data['label'] = enc_label
 num_col.append('label')

 corr_bin = bin_data[num_col].corr()
 num_col = list(multi_data.select_dtypes(include='number').columns)
 corr_multi = multi_data[num_col].corr()
 corr_ybin = abs(corr_bin['label'])
 highest_corr_bin = corr_ybin[corr_ybin >0.3]
 highest_corr_bin.sort_values(ascending=True)
 bin_cols = highest_corr_bin.index
 bin_data = bin_data[bin_cols].copy()
 bin_data.to_csv('bin_data.csv')

 corr_ymulti = abs(corr_multi['label'])
 highest_corr_multi = corr_ymulti[corr_ymulti >0.3]
 highest_corr_multi.sort_values(ascending=True)
 print(corr_ymulti[corr_ymulti >0.0])
 multi_cols = highest_corr_multi.index
 multi_data = multi_data[multi_cols].copy()
 multi_data.to_csv('multi_data.csv')