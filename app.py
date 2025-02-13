import streamlit as st
import openai-whisper
import os
# Use a pipeline as a high-level helper
from transformers import pipeline


# Load the Whisper model
model = openai-whisper.load_model("base")

st.title("Audio Transcriber")
st.write("Upload your audio file to get a transcription and audio summary.")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    # Save the uploaded file
    audio_path = "small.mp3"
    # with open(audio_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())

    st.write("Transcribing audio...")
    
    # Perform transcription
    result = model.transcribe(audio_path)
    transcript = result["text"]
    
    st.subheader("Transcription")
    st.write(transcript)

    # Summarize using GPT
    st.write("Generating meeting summary...")

    summarizer = pipeline("summarization", model="Falconsai/text_summarization")

    summary = summarizer(transcript, max_length=1000, min_length=30, do_sample=False)

    st.subheader("Meeting Summary")
    st.write(summary[0]["summary_text"])

    # Option to download transcript
    st.download_button("Download Transcript", transcript, "transcript.txt")
    st.download_button("Download Summary", summary[0]["summary_text"], "summary.txt")
