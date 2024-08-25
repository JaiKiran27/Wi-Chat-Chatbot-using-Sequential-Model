import tkinter as tk
from tkinter import scrolledtext
import spacy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import json
import random

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}")

# Load intents
intents_path = r'C:\Users\Asus\OneDrive\Desktop\Chatbot\intents.json'
try:
    with open(intents_path) as file:
        intents = json.load(file)
except Exception as e:
    print(f"Error loading intents file: {e}")

# Load model and data
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

# Tokenize sentences
def tokenize(sentence):
    doc = nlp(sentence)
    return [token.text.lower() for token in doc]

# Predict response
def predict_class(sentence):
    tokens = tokenize(sentence)
    bag_of_words = [1 if w in tokens else 0 for w in words]
    prediction = model.predict(np.array([bag_of_words]))[0]
    max_index = np.argmax(prediction)
    return classes[max_index]

# Generate response
def chatbot_response(message):
    tag = predict_class(message)
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            if "not resolved" in message.lower():
                response += "\nIf the issue persists, please contact support at wifive12345@gmail.com."
            return response
    return "Sorry, I didn't understand that. Please try again or contact support at wifive12345@gmail.com."

# Tkinter GUI
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

# Create window
root = tk.Tk()
root.title("Wi-Five Chatbot")
root.configure(bg="#000000")  # Set background color to black

# Create chat log
chat_log = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD, bg="#1a1a1a", fg="#FFFFFF", font=("Helvetica", 14))
chat_log.tag_configure('user', foreground='#FF6347')  # Tomato color for user text
chat_log.tag_configure('bot', foreground='#7FFF00')  # Chartreuse color for Wi-Chat text
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create input field
entry = tk.Entry(root, font=("Helvetica", 14), bg="#1a1a1a", fg="#FFFFFF", insertbackground="#FFFFFF")
entry.pack(padx=10, pady=10, fill=tk.X, expand=True)

# Create send button
send_button = tk.Button(root, text="Send", command=send_message, font=("Helvetica", 14), bg="#FF4500", fg="#FFFFFF", relief=tk.RAISED)
send_button.pack(padx=10, pady=10, side=tk.RIGHT)

# Bind Enter key to send message
root.bind('<Return>', send_message)

# Start the GUI
root.mainloop()
