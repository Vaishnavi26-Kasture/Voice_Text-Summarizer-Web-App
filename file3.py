import streamlit as st
import speech_recognition as sr
import sqlite3
import pandas as pd
from transformers import pipeline
from PIL import Image

# Load images
mic_image = Image.open("mic.jpeg")
summarize_image = Image.open("summarize.jpeg")

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Connect to SQLite database (or create it if it doesn’t exist)
conn = sqlite3.connect("transcriptions.db")
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcribed_text TEXT,
    summarized_text TEXT
)
""")
conn.commit()

# Streamlit UI
st.set_page_config(page_title="Voice Summarizer", page_icon="🎤", layout="centered")
st.title("🎤 Voice-to-Text & Summarization")

recognizer = sr.Recognizer()

if st.button("🎙️ Start Recording"):
    with sr.Microphone() as source:
        st.write("🎧 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)

    try:
        text_output = recognizer.recognize_google(audio_data)
        st.success("✅ Speech Transcribed Successfully!")
        st.text_area("📜 Transcribed Text:", text_output, height=150)

        # Summarization
        st.image(summarize_image, width=100)
        st.write("📝 Generating Summary...")
        summary = summarizer(text_output, max_length=50, min_length=20, do_sample=False, num_beams=4)
        summary_text = summary[0]["summary_text"]

        st.success("✅ Summarization Complete!")
        st.text_area("📄 Summarized Text:", summary_text, height=100)

        # Store in SQLite database
        cursor.execute("INSERT INTO transcriptions (transcribed_text, summarized_text) VALUES (?, ?)", 
                       (text_output, summary_text))
        conn.commit()
        st.success("✅ Data saved successfully!")

    except sr.UnknownValueError:
        st.error("⚠️ Speech not understood. Please try again.")
    except sr.RequestError:
        st.error("⚠️ Issue with the speech recognition service.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")

# Display stored transcriptions
st.subheader("📂 Stored Transcriptions")
df = pd.read_sql_query("SELECT * FROM transcriptions", conn)

if not df.empty:
    st.dataframe(df)

    # Download button for exporting data as CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Data", csv, "transcriptions.csv", "text/csv")
else:
    st.write("No stored transcriptions yet.")

st.markdown("🚀 **Developed with ❤️ using Streamlit & SQLite**")

# Close the database connection
conn.close()
