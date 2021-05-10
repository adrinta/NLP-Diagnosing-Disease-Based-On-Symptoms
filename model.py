import pandas as pd
import numpy as np
import xgboost as xgb
import nltk
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('dataset.csv')
ps = PorterStemmer()
def normalize(text):
    if text == text:
        text = text.lower()
        text = text.replace('_', ' ')
        text = ' '.join([ps.stem(word) for word in text.split()])
    else:
        text = ''
    return text
 
for i in range(1, len(df.columns)):
	df[df.columns[i]] = df[df.columns[i]].apply(lambda x: normalize(x))

cat_symptom = []
for symptom in df[df.columns[1:]].values:
    cat_symptom.append(' '.join(symptom).strip())
df['cat_symptom'] = cat_symptom

vectorizer = TfidfVectorizer()
le = LabelEncoder()
X = df['cat_symptom'].values
y = le.fit_transform(df[df.columns[0]].values)


X_train, X_test, y_train, y_test = train_test_split(X, y,
													random_state=88,
													stratify=y)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

classifier = xgb.XGBClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


print('accuracy score: {}'.format(accuracy_score(y_test, y_pred)))

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(le, open('le.pkl', 'wb'))
pickle.dump(classifier, open('model.pkl', 'wb'))