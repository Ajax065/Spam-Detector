import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #Vectorization
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Loading the data
try:
    print("Loading data...")
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print("Successful")
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please download the SMS Spam Collection dataset and place it in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()

#Dropping irrelevant Columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, errors='ignore')

#Renaming the column
df.columns = ['label', 'message']

# Convert 'ham' and 'spam' labels to 0 and 1
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data into features (X = messages) and target (y = labels)
X = df['message']
y = df['label_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Vectorising the text
print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

# Fit the vectorizer ONLY on the training data and then transform both sets
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)
print("Vectorized Successfully")

#Model Training
model = MultinomialNB()

#Tracking the time the model ran for
start_time = time.time()

#Training the model
model.fit(X_train_transformed, y_train)

end_time = time.time()

print(f"Model trained in {end_time - start_time:.2f} seconds.")

#Model testing and prediction
y_predict = model.predict(X_test_transformed)


#Metric Evaluation
print(f'Accuracy score is {accuracy_score(y_test,y_predict)*100:.2f} %')
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

#Testing the model
def predict_message(text_message):

    # Create a pandas Series from the single message (required for transform)
    new_message_series = pd.Series([text_message])
    
    # Transform the new message using the fitted vectorizer
    new_message_transformed = vectorizer.transform(new_message_series)
    
    # Predict the label
    prediction = model.predict(new_message_transformed)[0]

    # Get the probability scores (optional, but informative)
    probabilities = model.predict_proba(new_message_transformed)[0]
    
    label = 'SPAM' if prediction == 1 else 'HAM (Not Spam)'
    
    print("\n--- New Message Prediction ---")
    print(f"Message: '{text_message}'")
    print(f"Prediction: {label}")
    print(f"Confidence (Ham): {probabilities[0]:.4f}")
    print(f"Confidence (Spam): {probabilities[1]:.4f}")
    
    if label == 'SPAM':
        print("ALERT: This message is highly likely to be spam!")
        

# --- Testing with examples ---

# Example 1: Clear Spam
spam_example = "Congratulations! You have won a £1000 prize! Text CLAIM to 87021 to collect your reward."
predict_message(spam_example)

# Example 2: Clear Ham
ham_example = "Hey, do you want to grab lunch today? I'm free around 1 PM."
predict_message(ham_example)

#Example 3
ambiguous_example = "Urgent: Call 0800-0930-405 to claim your free mobile phone upgrade now."
predict_message(ambiguous_example)

#Example 4
AIESEC = "Hello there, thanks for your application to AIESEC, we really do apprecite the effort. Click the whatsapp link below to complete your process"
predict_message(AIESEC)