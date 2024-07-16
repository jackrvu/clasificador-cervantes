from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import pandas as pd
import langdetect
import re
# Example data (replace with your actual dataset)

mexican_posts = pd.read_csv("/Users/jackvu/Desktop/PDS/dialect_finder/venv/mexico_subreddit_posts.csv")["text"]
guatemalan_posts = pd.read_csv("/Users/jackvu/Desktop/PDS/dialect_finder/venv/guatemala_subreddit_posts.csv")["text"]
argentinean_posts = pd.read_csv("/Users/jackvu/Desktop/PDS/dialect_finder/venv/argentina_subreddit_posts.csv")["text"]
colombian_posts = pd.read_csv("/Users/jackvu/Desktop/PDS/dialect_finder/venv/colombia_subreddit_posts.csv")["text"]
salvadorean_posts = pd.read_csv("/Users/jackvu/Desktop/PDS/dialect_finder/venv/ElSalvador_subreddit_posts.csv")["text"]
spanish_posts = pd.read_csv("/Users/jackvu/Desktop/PDS/dialect_finder/venv/spain_subreddit_posts.csv")["text"]


def preprocess_text(text):
    # Lowercase the text
    text = str(text)
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Remove stop words
    return text


X = []
y = []

for post in mexican_posts:
    text = preprocess_text(post)
    if len(text) > 0:
        try:
            if langdetect.detect(text) == 'es':
                X.append(text)
                y.append("Mexico")
        except langdetect.LangDetectException:
            # If language detection fails, skip the post
            continue

for post in guatemalan_posts:
    text = preprocess_text(post)
    if len(text) > 0:
        try:
            if langdetect.detect(text) == 'es':
                X.append(text)
                y.append("Guatemala")
        except langdetect.LangDetectException:
            # If language detection fails, skip the post
            continue

for post in argentinean_posts:
    text = preprocess_text(post)
    if len(text) > 0:
        try:
            if langdetect.detect(text) == 'es':
                X.append(text)
                y.append("Argentina")
        except langdetect.LangDetectException:
            # If language detection fails, skip the post
            continue

for post in colombian_posts:
    text = preprocess_text(post)
    if len(text) > 0:
        try:
            if langdetect.detect(text) == 'es':
                X.append(text)
                y.append("Colombia")
        except langdetect.LangDetectException:
            # If language detection fails, skip the post
            continue

for post in salvadorean_posts:
    text = preprocess_text(post)
    if len(text) > 0:
        try:
            if langdetect.detect(text) == 'es':
                X.append(text)
                y.append("El Salvador")
        except langdetect.LangDetectException:
            # If language detection fails, skip the post
            continue

for post in spanish_posts:
    text = preprocess_text(post)
    if len(text) > 0:
        try:
            if langdetect.detect(text) == 'es':
                X.append(text)
                y.append("Castilian")
        except langdetect.LangDetectException:
            # If language detection fails, skip the post
            continue

# Create CountVectorizer and fit it to the data
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
joblib.dump(clf, 'spanish_dialect_prediction_reddit.pkl')
def find_probability(phrase):
    # Vectorize the input phrase using the trained CountVectorizer
    X_input = vectorizer.transform([preprocess_text(phrase)])

    # Predict probabilities using the trained classifier
    probabilities = clf.predict_proba(X_input)

    # Extract probability for Mexican class (index 1)
    probability_mexican = probabilities[0, 1]

    # Print the probabilities
    print(f"Probability of the phrase '{phrase}' being Mexican: {probability_mexican:.4f}")
    print(f"Probability of the phrase'{phrase}' being Guatemalan: {(1 - probability_mexican):.4f}")


input_phrases = [
    "¡Qué onda güey!",  # Mexican
    "No manches.",  # Mexican
    "¿Neta?",  # Mexican
    "Órale, vámonos.",  # Mexican
    "Está chido.",  # Mexican
    "¿Qué más?",  # Colombian
    "Parce, vamos a rumbear.",  # Colombian
    "Todo bien, todo bien.",  # Colombian
    "Estás tragado.",  # Colombian
    "¡Bacano!",  # Colombian
    "Che, ¿cómo andás?",  # Argentinean
    "Boludo, no me jodas.",  # Argentinean
    "¿Qué hacés?",  # Argentinean
    "Vamos a tomar unos mates.",  # Argentinean
    "¡Re copado!",  # Argentinean
    "¿Qué tal?",  # Castilian
    "Vale, perfecto.",  # Castilian
    "¡Qué guay!",  # Castilian
    "Estoy flipando.",  # Castilian
    "Tío, no te rayes.",  # Castilian
    "¿Cachai?",  # Chilean
    "¡Qué bacán!",  # Chilean
    "Está filete.",  # Chilean
    "Vamos a la pega.",  # Chilean
    "¿Cómo estai?",  # Chilean
    "No me gusta la economía aquí en España"
]


X_input = vectorizer.transform(input_phrases)

# Predict probabilities for each input phrase
probabilities = clf.predict_proba(X_input)

# Get the list of unique dialects from the classifier
unique_dialects = clf.classes_

# Print the probabilities for each input phrase and each dialect
for i, phrase in enumerate(input_phrases):
    print(f"Phrase: '{phrase}'")
    for j, dialect in enumerate(unique_dialects):
        print(f"Probability of '{dialect}': {probabilities[i, j]:.4f}")
    print()  # Print empty line for readability

