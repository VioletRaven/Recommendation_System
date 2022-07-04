#user data --> age, gender, military grade, other anagraphical records
#content/rating data --> category, recurrent terms, document length, previous download history
# if pdf --> probability of download
# if html --> probability of opening the page from n of open links and all same features

'''
build toy dataset of users
'''
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

gradi = ['generale', 'colonnello', 'tenente colonnello',
         'maggiore', 'capitano', 'tenente', 'sottotenente',
         'luogotenente', 'maresciallo', 'sergente',
         'primo aviere', 'aviere']

distribution = np.floor(abs(skewnorm.rvs(10, size = 13) * 45))
distribution

tutti_gradi = []
for g, d in zip(gradi, distribution):
    print(g, d)
    tutti_gradi.append([g] * np.int(d))

gradi_distribuiti = [item for sublist in tutti_gradi for item in sublist]
gradi_distribuiti

utente_id = np.abs(np.floor(np.random.normal(50, 25, len(gradi_distribuiti))))
random.shuffle(utente_id)
age = np.abs(np.floor(np.random.normal(40, 20, len(gradi_distribuiti)))) # --> not needed


l = [random.randint(0,10) for i in range(len(gradi_distribuiti))]
gender = [0 if x <= 5 else 1 for x in l]
gender

tipi_di_pdf = ['romantico', 'azione', 'thriller', 'fantasy', 'sci-fi', 'fantascientifico', 'western'] * 57
random.shuffle(tipi_di_pdf)

categorie = ['Area Giuridico-Legale', 'Attività Finanziaria', 'Informazioni e Sicurezza', 'Tutela della Salute'] * 99
categorie_plus = ['Area Giuridico-Legale', 'Attività Finanziaria', 'Informazioni e Sicurezza'] # per arrivare a 399 (cambia ogni volta per ora)
categorie = categorie + categorie_plus
random.shuffle(categorie)

toy_df = pd.DataFrame(zip(gradi_distribuiti, age, utente_id, gender, tipi_di_pdf, categorie), columns = ['Gradi', 'Eta', 'ID', 'Sesso', 'tipi_di_pdf', 'categorie'])

from random import randrange
from datetime import timedelta, datetime

def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)
d1 = datetime.strptime('1/1/1950', '%m/%d/%Y')
d2 = datetime.strptime('1/1/2003', '%m/%d/%Y')
date = random_date(d1, d2)
def get_dob(dob):
    day = dob.day
    month = dob.month
    year = dob.year
    return '{}/{}/{}'.format(day, month, year)

def get_year(dob):
    year = dob.year
    return int(year)


#toy_df.to_csv(r'C:\Users\Andrea\Desktop\toy_dataset.txt')

df = pd.read_csv(r'C:\Users\Andrea\Desktop\toy_dataset.txt', index_col = False)
df = df.drop('Unnamed: 0', axis = 1)

categorical_cols = ['Gradi','tipi_di_pdf']
df = pd.get_dummies(df, columns = categorical_cols)

y = df[['categorie']]
y = y.values
df = df.drop('categorie', axis = 1) #remove categorie
X = df.values

# encode Y class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(np.ravel(y))
label_encoded_y = label_encoder.transform(np.ravel(y))
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)
# fit model no training data

# train, test = train_test_split(df, test_size=0.2, random_state=50)
#
# y_train = train[['categorie']]
# X_train = train.loc[:, train.columns != 'categorie']
#
# y_test = test[['categorie']]
# X_test = test.loc[:, test.columns != 'categorie']
'''
'''

#Recommendation model setting

model1 = xgb.XGBRegressor(objective='reg:squarederror')
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)
rmse = np.sqrt(np.mean((pred1 - y_test)**2))
print(f'content-based rmse = {rmse}')

pred1_rounded = [int(round(x, 0)) for x in pred1]
estimations = [1 if x == c else 0 for x, c in zip(pred1_rounded, y_test)]
( sum(estimations) / len(estimations) )* 100

# simple classifier
from sklearn.metrics import accuracy_score

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

'''
different try
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r'C:\Users\Andrea\Desktop\toy_dataset.txt', index_col = False)
df = df.drop('Unnamed: 0', axis = 1)

df["combined_features"] = df['Gradi'] + " " + df['tipi_di_pdf'] + " " + df['categorie']
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
print("Count Matrix:", count_matrix.toarray())
cosine_sim = cosine_similarity(count_matrix)

'''
examples
'''
his_grade = "maggiore"
def get_index_from_title(title):
    return df[df.Gradi == title].values[0]
movie_index = get_index_from_title(his_grade)

movie_user_likes = "western"
def get_index_from_title(title):
    return df[df.tipi_di_pdf == title].values[0]
movie_index = get_index_from_title(movie_user_likes)

df.set_index('ID', inplace=True)
movie_index = df.loc[(df['tipi_di_pdf'] == 'western') & (df['Gradi'] == 'maggiore')].values[0]

similar_movies = list(enumerate(cosine_sim[movie_index]))
