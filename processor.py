from dotenv import load_dotenv
import openai
import requests
import json  
import os
from gtts import gTTS

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

class Processor:

    def __init__(self):
        self.AUDIO_PATH = 'res.wav'
        self.LANGUAGE = 'en'
        self.GRAMMAR_PROMPT_PREPEND = 'correct grammar : \n'
        self.WHISPER_API_URL = 'https://whisper.lablab.ai/asr'
        self.error_conversation_log = []
        self.conversation_log = ['The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.']

    def get_last_response(self):
        return self.conversation_log[-1]

    def fix_grammar(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=self.GRAMMAR_PROMPT_PREPEND + prompt,
            temperature=0.5,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response['choices'][0]['text']

    def generate_answer(self):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt='\n'.join(self.conversation_log),
            temperature=0.5,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response['choices'][0]['text']

    def generate_transcript(self, audio_path):
        files=[
            ('audio_file',('test1.mp3',open(audio_path,'rb'),'audio/mpeg'))
        ]
        response = requests.request("POST", self.WHISPER_API_URL, data={}, files=files)
        response = json.loads(response.text)
        return response['text']

    def generate_audio(self):
        obj = gTTS(text=self.get_last_response(), lang=self.LANGUAGE, slow=False)
        obj.save(self.AUDIO_PATH)
        return self.AUDIO_PATH

    def get_conversation_log(self, error=False):
        log = self.error_conversation_log if error else self.conversation_log[1:]
        prepended_log = ['YOU: ' + log[i] if i % 2 == 0 else 'CAMRI: ' + log[i] for i in range(len(log))]
        return '\n'.join(prepended_log)

    def run(self, audio_path):
        transcript = self.generate_transcript(audio_path)
        self.error_conversation_log.append(transcript)
        
        fixed_transcript = self.fix_grammar(transcript)
        self.conversation_log.append(fixed_transcript)

        bot_answer = self.generate_answer()
        self.conversation_log.append(bot_answer)
        self.error_conversation_log.append(bot_answer)

        return self.generate_audio()
