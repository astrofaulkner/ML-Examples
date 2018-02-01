import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# load pima indians dataset
ind = pd.read_csv('pima-indians-diabetes.csv',
                  names = ['f%d' % x for x in range(1,9)]+['label'])
#print(ind.head())
print(ind.describe().transpose())

# separate features from labels
X = ind.drop('label', axis=1)
y = ind['label']
#print(X.head())
#print(y.head())

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.describe().transpose())
print(X_test.describe().transpose())

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

# evaluate the model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
