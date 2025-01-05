
import os
import csv
import datetime
import nltk
import ssl
import streamlit as st
import random
import json
import speech_recognition as sr
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Handle SSL context (Only needed in some environments)
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK data handling (Do this ONCE outside of Streamlit)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)
try:
 nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Load intents (with robust error handling)
try:
    with open('elderly_intents.json', 'r', encoding='utf-8') as f:
        try:
            intents = json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in elderly_intents.json: {e}")
            st.stop()
except FileNotFoundError:
    st.error("elderly_intents.json not found! Please create it.")
    st.stop()

# Data preprocessing and model training (done ONCE)
tags = []
patterns = []
for intent in intents.get('intents', []):
    for pattern in intent.get('patterns', []):
        tags.append(intent['tag'])
        patterns.append(pattern)

if patterns:
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(patterns)
    y = tags

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=42, max_iter=10000)
    clf.fit(X_train, y_train)

    # Model Evaluation
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    def chatbot(input_text):
        input_text = vectorizer.transform([input_text])
        tag = clf.predict(input_text)[0]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                responses = intent.get('responses', ["I'm not sure I understand."])
                if intent['tag'] == "date_time":
                    now = datetime.datetime.now()
                    formatted_responses = [
                        response.format(now.strftime("%Y-%m-%d %H:%M:%S")) for response in responses
                    ]
                    return random.choice(formatted_responses)
                return random.choice(responses)
        return "I'm not sure I understand."
else:
    st.error("No patterns found in elderly_intents.json. Please add some intents and patterns.")
    st.stop()

# Voice Input Function
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your voice...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            user_input = recognizer.recognize_google(audio)
            st.write(f"You said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand that. Please try again.")
        except sr.RequestError as e:
            st.write(f"Error with the speech recognition service: {e}")
    return None

# Text-to-Speech Function
def speak_response(response_text):
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")
    os.system("start response.mp3")  # Use 'start' for Windows, 'afplay' for Mac, or 'mpg123' for Linux.

# Streamlit app
def main():
    st.title("Elderly Assistant Chatbot with Voice Interaction")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! How can I assist you today?")
        st.write("Type your query below or use voice input.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Voice Input Button
        if st.button("Use Voice Input"):
            user_input = get_voice_input()
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                bot_response = chatbot(user_input)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
                speak_response(bot_response)

        # Text Input Chat
        if user_input := st.chat_input("You:"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            bot_response = chatbot(user_input)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            speak_response(bot_response)

        if st.session_state.messages and st.session_state.messages[-1]["content"].lower() in ['goodbye', 'bye']:
            st.write("You're welcome! Have a great day!")
            st.session_state.messages.clear()

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history yet.")

    elif choice == "About":
        st.write("This chatbot is designed to assist elderly users with daily tasks and information.")
        st.subheader("Features:")
        st.write("- Reminders (medication, appointments)")
        st.write("- Basic information retrieval (weather, date/time)")
        st.write("- Simple conversation")
        st.subheader("Future Enhancements:")
        st.write("- Personalized profiles")
        st.write("- Integration with emergency services")

if __name__ == '__main__':
    main()
