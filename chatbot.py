import torch
from transformers import pipeline
import tkinter as tk
import speech_recognition as sr
import pyttsx3

# Load GPT-2 model
chatbot = pipeline("text-generation", model="gpt2")

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Function to recognize speech (Uses Google Web Speech API or Vosk for offline)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        chat_log.insert(tk.END, "Listening...\n")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            user_input = recognizer.recognize_google(audio)  # Use "vosk" for offline
            chat_log.insert(tk.END, f"You: {user_input}\n")
            get_gpt_response(user_input)
        except sr.UnknownValueError:
            chat_log.insert(tk.END, "Could not understand. Please try again.\n")
        except sr.RequestError:
            chat_log.insert(tk.END, "Speech recognition service error.\n")

# Function to get GPT-2 response
def get_gpt_response(user_input):
    response = chatbot(user_input, max_length=50, num_return_sequences=1)
    bot_reply = response[0]["generated_text"]
    chat_log.insert(tk.END, f"Bot: {bot_reply}\n")
    
    # Speak the response
    engine.say(bot_reply)
    engine.runAndWait()

# GUI Setup with Tkinter
root = tk.Tk()
root.title("Offline Voice Chatbot")

# Chat Log Display
chat_log = tk.Text(root, height=15, width=50)
chat_log.pack()

# Voice Input Button
voice_btn = tk.Button(root, text="Speak", command=recognize_speech)
voice_btn.pack()

# Run the GUI
root.mainloop(
