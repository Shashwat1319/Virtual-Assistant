from flask import Flask, render_template, jsonify
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
from groq import Groq
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np

load_dotenv()  # Load variables from .env

app = Flask(__name__)

# Initialize Groq client with API key (set this securely in env or paste here)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Or replace with your actual API key as string

def speak(text):
    """Function to speak the given text."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

def run_groq(prompt):
    """Function to interact with Groq's API for AI responses."""
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gemma2-9b-it",  # Change the model name based on your needs
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
        )

        ai_response = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                ai_response += content
        return ai_response.strip()

    except Exception as e:
        print(f"[Groq Error] {e}")
        return "Sorry, I couldn't connect to the AI service."

def process_command(command):
    """Process user command and return an appropriate response."""
    command = command.lower()
    if 'hey' in command or 'hello' in command:
        return "Hello, how may I help you?"
    elif 'bye' in command:
        return "Bye, have a nice day!"
    elif 'open google' in command:
        webbrowser.open("https://google.com")
        return "Opening Google..."
    elif 'open youtube' in command:
        webbrowser.open("https://youtube.com")
        return "Opening YouTube..."
    elif 'time' in command:
        return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}"
    elif 'date' in command:
        return f"Today's date is {datetime.datetime.now().strftime('%A, %B %d, %Y')}"
    else:
        return run_groq(command)

@app.route("/")
def home():
    """Home route that renders the index page."""
    return render_template("index.html")

@app.route("/listen")
def listen():
    """Route to listen to the user's speech and process it."""
    recognizer = sr.Recognizer()
    try:
        # Use sounddevice to record audio
        fs = 16000  # Sample rate for recording
        duration = 5  # Duration of the recording in seconds
        print("🎧 Listening from frontend...")

        # Record audio using sounddevice
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to finish

        # Convert numpy array to an audio file-like object
        audio_data = np.array(audio_data, dtype=np.int16)

        # Recognize the audio using Google speech recognition
        audio = sr.AudioData(audio_data.tobytes(), fs, 2)  # 2 channels for stereo audio
        user_input = recognizer.recognize_google(audio)
        print("User said:", user_input)

        response = process_command(user_input)
        speak(response)

        return jsonify({"user": user_input, "response": response})
    
    except Exception as e:
        print("⚠️ Error in /listen:", str(e))
        error_response = "Sorry, I didn't catch that."
        speak(error_response)
        return jsonify({"user": "Unrecognized", "response": error_response})

if __name__ == "__main__":
    app.run(debug=True)
