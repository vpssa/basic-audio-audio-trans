from pydub import AudioSegment
import torch
import os
from TTS.api import TTS

class TTSGenerator:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        self.tts = TTS(model_name, gpu=torch.cuda.is_available())
        
    def generate_audio(self, sentences, output_file="final_merged_output.wav", speaker_ref=None, pause_duration=500):
        output_dir = os.path.dirname(output_file) or "."
        os.makedirs(output_dir, exist_ok=True)
        
        final_audio = AudioSegment.silent(duration=0)
        silence = AudioSegment.silent(duration=pause_duration)
        
        for i, sentence in enumerate(sentences):
            temp_file = os.path.join(output_dir, f"temp_{i}.wav")
            self.tts.tts_to_file(
                text=sentence,
                file_path=temp_file,
                speaker_wav=speaker_ref,
                language="hi"
            )
            final_audio += AudioSegment.from_wav(temp_file) + silence
            os.remove(temp_file)
        
        final_audio = final_audio[:-pause_duration]  # Remove last pause
        final_audio.export(output_file, format="wav")
        return output_file