import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# load dataset
df = pd.read_csv('MOCK_DATA.csv', skiprows=1,
                  names = ['id', 'md5', 'name', 'timestamp_1', 'timestamp_2'])
df['timestamp_1'] = pd.to_datetime(df['timestamp_1'])
df['timestamp_2'] = pd.to_datetime(df['timestamp_2'])
#print(df.head())

# Add additional column with binary value (0, 1) to test classification
# 0: Pass; 1: Fail
#df['result'] = np.random.randint(0, 2, df.shape[0])
#print(df.head())

# Add file size column
df['size'] = np.random.randint(0, 100e9, df.shape[0])

# Create separate 'root' and 'ext' columns for files
df['ext'] = 0
df['root'] = 0
df['name_len'] = 0
for i in range(len(df.index)):
    name = df['name'][i].split('.')
    name_s = len(name)
    if name_s == 1:
        df['ext'][i] = ''
        df['root'][i] = name[0]
    else:
        df['ext'][i] = name[name_s-1]
        df['root'][i] = '.'.join(name[:name_s-1])
    df['name_len'][i] = len(df['root'][i])
#print(df.head())

# Get time delta
df['delta_t'] = ((df.timestamp_2 - df.timestamp_1).astype('timedelta64[s]')).astype('int64')

# Generate a mapping between file extensions and integers
exts = sorted(df['ext'].unique())
exts_map = dict(zip(exts, range(0, len(exts) + 1)))
df['ext_val'] = df['ext'].map(exts_map).astype(int)
#print(df.head())

# Generate a mapping between file names and integers
nams = sorted(df['root'].unique())
nams_map = dict(zip(nams, range(0, len(nams) + 1)))
df['name_val'] = df['root'].map(nams_map).astype(int)
#print(df.head())

# Generate a mapping between file hashes and integers
md5s = sorted(df['md5'].unique())
md5s_map = dict(zip(md5s, range(0, len(md5s) + 1)))
df['md5_val'] = df['md5'].map(md5s_map).astype(int)
#print(df.head())

# Apply 'value' logic
df['result'] = 0
## Assign value if extension contains letter 'x'
##df.loc[df['ext'].str.contains('x'), 'result'] = 1
## Assign value if extension in list, 'size' > threshold, 'delta_t' < threshold, and 'md5_val' is in first half
df.loc[(df['ext_val'] < int(len(exts)/2.))
       #& (df['ext'].isin(['avi', 'bmp', 'eps', 'ima', 'imap', 'jpe', 'jpeg', 'jpg', 'mime', 'mov', 'mp2', 'mp3',
       #                  'mpe', 'mpeg', 'mpg', 'mpga', 'p10', 'p12', 'pic', 'pict', 'png', 'ppt', 'rgb', 'wav',
       #                  'word', 'x-png', 'xif']))
       & (df['size'] >= 1e6)
       & (df['delta_t'] < (0.5*df['delta_t'].max()+0.5*df['delta_t'].min()))
       & (df['md5_val'] < int(len(md5s)/2.))
       , 'result'] = 1
print('This many valued files: %d' % len(df[df['result'] == 1]))


# separate features from labels
X = df.drop(['result', 'id', 'md5', 'name', 'timestamp_1', 'timestamp_2',
             'ext', 'root'], axis=1)
y = df['result']
#print(X.head())
#print(y.head())

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

"""
print('\nInput sets:')
print(X_train.describe().transpose())
print(X_test.describe().transpose())

print('\nOutput sets:')
print(y_train.describe().transpose())
print(y_test.describe().transpose())
"""

# normalize data (both test and train sets)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# create model
mlp = MLPClassifier(hidden_layer_sizes=(12,8,1), batch_size=10, max_iter=150)

# fit model
mlp.fit(X_train, y_train)

# calculate predictions
predictions = mlp.predict(X_test)
print(predictions)
print(y_test)

# evaluate the model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
