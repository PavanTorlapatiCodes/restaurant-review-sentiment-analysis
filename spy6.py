import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Dataset Loading
dataset = pd.read_csv(r"C:\Users\T pavan kumar\Downloads\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# 2. Text Preprocessing
corpus = []
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]).lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    corpus.append(' '.join(review))

# 3. Vectorization
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 5. Best Fit Model 1: Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

print("--- Logistic Regression ---")
print("Train Score (Bias):", log_model.score(X_train, y_train))
print("Test Score (Variance):", log_model.score(X_test, y_test))

# 6. Best Fit Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
rf_model.fit(X_train, y_train)

print("\n--- Random Forest (Tuned) ---")
print("Train Score (Bias):", rf_model.score(X_train, y_train))
print("Test Score (Variance):", rf_model.score(X_test, y_test))
