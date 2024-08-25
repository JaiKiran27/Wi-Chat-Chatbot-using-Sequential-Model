# Wi-Chat-Chatbot-using-Sequential-Model

This code is a comprehensive implementation of a chatbot using machine learning with TensorFlow and Keras. The chatbot is trained to understand different patterns of user input and respond accordingly based on predefined intents.

Let’s break down the code step by step:

### 1. **Imports and Setup**
```python
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
```
- **spacy**: Used for natural language processing (tokenization).
- **json**: For handling the intents data, which is stored in a JSON format.
- **numpy**: Useful for handling arrays and performing mathematical operations.
- **tensorflow/keras**: Used to build and train the neural network.
- **sklearn**: Provides utilities for preprocessing, splitting data, and generating metrics.

### 2. **Loading spaCy Model**
```python
nlp = spacy.load("en_core_web_sm")
```
- The `spaCy` model is loaded to tokenize the text. `en_core_web_sm` is a small English model.

### 3. **Loading Intents Data**
```python
intents_path = r'C:\Users\Asus\OneDrive\Desktop\Chatbot\intents.json'
with open(intents_path) as file:
    intents = json.load(file)
```
- The JSON file containing various intents, patterns, and responses is loaded. Each intent contains patterns (examples of what the user might say) and the corresponding responses.

### 4. **Tokenization Function**
```python
def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text.lower() for token in doc]
```
- This function takes a sentence and returns a list of lowercase words (tokens) using the spaCy model.

### 5. **Processing Intents**
```python
classes = []  # List to hold all tags
documents = []  # List to hold tokenized patterns and their tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        documents.append((words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

classes.sort()
words = sorted(list(set(w for doc in documents for w in doc[0])))
```
- **classes**: This list holds all unique tags (intents).
- **documents**: This list stores each pattern's tokenized words along with the associated tag (intent).
- **words**: This is a sorted list of all unique words that appear in the patterns.

### 6. **Preparing Training Data**
```python
training_sentences = []
training_labels = []

for doc in documents:
    bag_of_words = [1 if w in doc[0] else 0 for w in words]
    training_sentences.append(bag_of_words)
    training_labels.append(classes.index(doc[1]))
```
- **Bag of Words**: Each pattern is converted into a numerical vector using the "bag of words" technique. If a word in the vocabulary is present in the pattern, the corresponding position in the vector is set to 1; otherwise, it's set to 0.
- **training_sentences**: List of these vectors for each pattern.
- **training_labels**: List of labels (tags) corresponding to each pattern.

### 7. **Splitting the Data**
```python
X_train, X_test, y_train, y_test = train_test_split(
    training_sentences, training_labels, test_size=0.2, random_state=42
)
```
- The data is split into training (80%) and testing (20%) sets.

### 8. **Building the Neural Network**
```python
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(len(classes), activation='softmax'))
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
```
- **Sequential model**: A linear stack of layers.
- **Dense layers**: Fully connected layers with ReLU activation. The first layer has 128 units, the second has 64.
- **Dropout**: Regularization technique to prevent overfitting by randomly dropping 40% of the neurons during training.
- **Output layer**: Uses a softmax activation function to output probabilities for each class.
- **Compile**: The model is compiled using sparse categorical cross-entropy as the loss function and Adam as the optimizer.

### 9. **Learning Rate Scheduler and Early Stopping**
```python
def scheduler(epoch, lr):
    if epoch > 50:
        return lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
```
- **LearningRateScheduler**: Decreases the learning rate by half after 50 epochs.
- **EarlyStopping**: Stops training when the validation loss doesn’t improve for 15 epochs, preventing overfitting and restoring the best weights.

### 10. **Training the Model**
```python
history = model.fit(
    np.array(X_train), np.array(y_train),
    epochs=200,
    batch_size=8,
    validation_data=(np.array(X_test), np.array(y_test)),
    callbacks=[early_stopping, lr_scheduler],
    verbose=2
)
```
- The model is trained with a batch size of 8 for a maximum of 200 epochs, using early stopping and the learning rate scheduler.

### 11. **Saving the Model and Data**
```python
model.save('chatbot_model.keras')
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
```
- The trained model is saved in a `.keras` file, and the `classes` and `words` lists are saved using `pickle` for later use.

### 12. **Evaluating the Model**
```python
test_loss, test_acc = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
```
- The model’s accuracy on the test set is printed.

### 13. **Generating Predictions and Classification Report**
```python
y_pred = np.argmax(model.predict(np.array(X_test)), axis=-1)
unique_labels = np.unique(y_test)
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=[classes[i] for i in unique_labels]))
```
- **y_pred**: The model's predictions are generated.
- **Classification Report**: A detailed report (precision, recall, F1-score) is generated for each class.

### Summary:
- **Purpose**: This script builds, trains, and evaluates a neural network model to classify user inputs based on predefined intents. The chatbot can then respond appropriately based on these classifications.
- **Model**: A neural network with two hidden layers, using regularization (Dropout and L2), early stopping, and a learning rate scheduler.
- **Data Preparation**: Text is tokenized and converted into a bag-of-words representation for training.
- **Evaluation**: The model's performance is evaluated using accuracy and a classification report. 

This setup allows the chatbot to learn from a relatively small dataset, generalize well, and avoid overfitting.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This code creates a simple chatbot application using `tkinter` for the GUI and TensorFlow/Keras for natural language processing. Here's a detailed explanation of each part:

### 1. **Imports and Setup**
```python
import tkinter as tk
from tkinter import scrolledtext
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import json
import random
```
- **`tkinter`**: Used for creating the GUI.
- **`spacy`**: For natural language processing (NLP) to tokenize sentences.
- **`numpy`**: For handling arrays and numerical operations.
- **`tensorflow`** and **`tensorflow.keras`**: For loading and using a pre-trained model.
- **`pickle`**: For loading Python objects saved to disk.
- **`json`**: For handling JSON data.
- **`random`**: For selecting random responses from a list.

### 2. **Load SpaCy Model**
```python
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
```
- Loads the spaCy model for tokenizing input text. If there is an error, it prints an error message.

### 3. **Load Intents, Model, and Data**
```python
intents_path = r'C:\Users\Asus\OneDrive\Desktop\Chatbot\intents.json'
try:
    with open(intents_path) as file:
        intents = json.load(file)
except Exception as e:
    print(f"Error loading intents file: {e}")

model_path = r'C:\Users\Asus\OneDrive\Desktop\Chatbot\chatbot_model.keras'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

classes_path = r'C:\Users\Asus\OneDrive\Desktop\Chatbot\classes.pkl'
try:
    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)
except Exception as e:
    print(f"Error loading classes file: {e}")

words_path = r'C:\Users\Asus\OneDrive\Desktop\Chatbot\words.pkl'
try:
    with open(words_path, 'rb') as f:
        words = pickle.load(f)
except Exception as e:
    print(f"Error loading words file: {e}")
```
- **`intents.json`**: Contains the intents and corresponding responses.
- **`chatbot_model.keras`**: The pre-trained model used for predicting responses.
- **`classes.pkl`**: Contains class labels for different intents.
- **`words.pkl`**: Contains the vocabulary used for tokenization and bag-of-words representation.

### 4. **Tokenization and Prediction Functions**
```python
def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text.lower() for token in doc]

def predict_class(sentence):
    tokens = tokenize(sentence)
    bag_of_words = [1 if w in tokens else 0 for w in words]
    prediction = model.predict(np.array([bag_of_words]))[0]
    max_index = np.argmax(prediction)
    return classes[max_index]

def chatbot_response(message):
    tag = predict_class(message)
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            if "not resolved" in message.lower():
                response += "\nIf the issue persists, please contact support at wifive12345@gmail.com."
            return response
    return "Sorry, I didn't understand that. Please try again or contact support at wifive12345@gmail.com."
```
- **`tokenize`**: Converts a sentence into a list of lowercase tokens using spaCy.
- **`predict_class`**: Converts tokens into a bag-of-words representation and uses the model to predict the class (intent) of the sentence.
- **`chatbot_response`**: Uses the predicted intent to fetch a response from the intents data. It also provides a contact support message if the user indicates their issue isn’t resolved.

### 5. **Tkinter GUI Setup**
```python
def send_message(event=None):
    user_message = entry.get()
    if user_message:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, "User: " + user_message + '\n', 'user')
        response = chatbot_response(user_message)
        chat_log.insert(tk.END, "Wi-Chat: " + response + '\n', 'bot')
        chat_log.config(state=tk.DISABLED)
        entry.delete(0, tk.END)
        chat_log.yview(tk.END)  # Auto-scroll to the bottom

root = tk.Tk()
root.title("Wi-Five Chatbot")
root.configure(bg="#000000")  # Set background color to black

chat_log = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD, bg="#1a1a1a", fg="#FFFFFF", font=("Helvetica", 14))
chat_log.tag_configure('user', foreground='#FF6347')  # Tomato color for user text
chat_log.tag_configure('bot', foreground='#7FFF00')  # Chartreuse color for Wi-Chat text
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(root, font=("Helvetica", 14), bg="#1a1a1a", fg="#FFFFFF", insertbackground="#FFFFFF")
entry.pack(padx=10, pady=10, fill=tk.X, expand=True)

send_button = tk.Button(root, text="Send", command=send_message, font=("Helvetica", 14), bg="#FF4500", fg="#FFFFFF", relief=tk.RAISED)
send_button.pack(padx=10, pady=10, side=tk.RIGHT)

root.bind('<Return>', send_message)

root.mainloop()
```
- **`send_message`**: Handles sending the user message and displaying the chatbot’s response in the chat log. It also auto-scrolls to the bottom of the chat log.
- **`root`**: Main window for the GUI. 
- **`chat_log`**: Displays the conversation between the user and the chatbot.
- **`entry`**: Input field for the user to type their message.
- **`send_button`**: Button to send the message.
- **`root.bind('<Return>', send_message)`**: Allows sending the message by pressing the Enter key.
- **`root.mainloop()`**: Starts the Tkinter event loop to display the window.

This code sets up a basic chatbot interface where users can type messages, receive responses, and see the conversation history.
