"""
Utility script to list all available Gemini models supporting 'generateContent'
for the configured GEMINI_API_KEY.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY is not set. Please check your .env file.")
else:
    genai.configure(api_key=api_key)
    print("Available Gemini models for text generation:")
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f" - {m.name}")
    except Exception as e:
        print(f"Error fetching models: {e}")
