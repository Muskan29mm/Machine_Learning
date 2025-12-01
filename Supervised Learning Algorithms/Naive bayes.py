# Implementation of Naive Bayes Classifier using scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love this movie", "This film was terrible", "Amazing experience",
         "Worst movie ever", "I enjoyed it", "Not good at all"]

labels = [1,0,1,0,1,0] # 1: Positive, 0: Negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=20)

model=  MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, pred))
