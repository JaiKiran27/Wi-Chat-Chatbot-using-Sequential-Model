import spacy
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_sm")

# Load intents from the JSON file
intents_path = r'C:\Users\Asus\OneDrive\Desktop\Chatbot\intents.json'
with open(intents_path) as file:
    intents = json.load(file)

# Function to tokenize a sentence into lowercase words
def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text.lower() for token in doc]

# Process intents: extract patterns and associate them with tags
classes = []  # List to hold all tags
documents = []  # List to hold tokenized patterns and their tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        documents.append((words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Sort classes alphabetically for consistent indexing
classes.sort()

# Create a sorted list of all unique words from the patterns
words = sorted(list(set(w for doc in documents for w in doc[0])))

# Prepare training data
training_sentences = []
training_labels = []

for doc in documents:
    # Create a bag of words representation for each pattern
    bag_of_words = [1 if w in doc[0] else 0 for w in words]
    training_sentences.append(bag_of_words)
    training_labels.append(classes.index(doc[1]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    training_sentences, training_labels, test_size=0.2, random_state=42
)

# Build the neural network model with improved regularization and adjusted learning rate
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.4))  # Adjusted Dropout rate
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.4))  # Adjusted Dropout rate
model.add(Dense(len(classes), activation='softmax'))
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Learning rate scheduler to decrease learning rate during training
def scheduler(epoch, lr):
    if epoch > 50:
        return lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# Define early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increased patience

# Train the model with early stopping and learning rate scheduler
history = model.fit(
    np.array(X_train), np.array(y_train),
    epochs=200,
    batch_size=8,
    validation_data=(np.array(X_test), np.array(y_test)),
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)

# Save the trained model and the data (classes and words)
model.save('chatbot_model.keras')
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

# Optionally, evaluate the model on the test set and print additional metrics
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Obtain predictions from the model
y_pred = np.argmax(model.predict(np.array(X_test)), axis=-1)

# Ensure classes and labels are correctly matched
unique_labels = np.unique(y_test)
print("Number of classes in model:", len(classes))
print("Number of unique labels in y_test:", len(unique_labels))
print("Unique labels in y_test:", unique_labels)
print("Unique labels in y_pred:", np.unique(y_pred))

# Generate the classification report
try:
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=[classes[i] for i in unique_labels]))
except ValueError as e:
    print("Error in classification_report:", e)
    print("Number of classes:", len(classes))
    print("Target names length:", len(classes))
