# D:\AI\Gits\hebrew-tutor-data-pipeline\test_speech.py
import os
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file
import azure.cognitiveservices.speech as speechsdk
speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_SPEECH_KEY'), region=os.getenv('AZURE_SPEECH_REGION'))
print("Speech service connected successfully!")