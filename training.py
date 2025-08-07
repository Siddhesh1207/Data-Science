import pandas as pd
import re

df = pd.read_csv('Nike_dataset.csv')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['Cleaned_Reviews'] = df['Reviews_posted'].apply(clean_text)

# Remove empty reviews
df = df[df['Cleaned_Reviews'].str.strip() != '']

print("Sample cleaned reviews:")
print(df['Cleaned_Reviews'].head())

# Now continue with vectorization and training as before...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Assuming df already loaded and cleaned, with 'Cleaned_Reviews' and 'Rating' columns

# 1. Prepare features and labels
X = df['Cleaned_Reviews']
y = df['Rating']

# 2. Split dataset into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train the Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test_tfidf)

# 6. Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
import pickle

# Save the trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open('model.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)
