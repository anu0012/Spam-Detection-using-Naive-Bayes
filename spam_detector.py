import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read data as table and give lables to columns
df = pd.read_table("smsspamcollection/SMSSpamCollection",
					sep='\t',
					header=None,
					names=['label','sms_content'])


# Give numerical values to labels
df['label'] = df.label.map({'ham':0, 'spam':1})

# Split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(df['sms_content'],
													df['label'],
													random_state=1)

# CountVectorizer to create BagOfWords (By default it will remove all stopwords,
# change each word into lowercase and it won't consider punctuation)
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix.
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)

predictions = naive_bayes.predict(testing_data)

# Model Evaluation

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))









