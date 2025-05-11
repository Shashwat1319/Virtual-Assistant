
from flask import Flask, render_template, jsonify
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # load variables from .env
app = Flask(__name__)

# Initialize Groq client with API key (set this securely in env or paste here)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 # or replace with your actual API key as string

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

def run_groq(prompt):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gemma2-9b-it",
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
    return render_template("index.html")

@app.route("/listen")
def listen():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎧 Listening from frontend...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

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
