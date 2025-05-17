from flask import Flask, request, render_template, send_file
import os
from utils.downloader import download_and_transcribe
from utils.sentence_splitter import split_into_sentences
from utils.translator import Translator
from utils.tts_generator import TTSGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Initialize models at startup
translator = Translator()
tts = TTSGenerator()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form['url']
        speaker_ref = request.files.get('speaker_ref')
        
        try:
            # Process input
            transcript_path = download_and_transcribe(video_url, app.config['UPLOAD_FOLDER'])
            
            with open(transcript_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            sentences = split_into_sentences(text)
            translations = translator.translate_batch(sentences)
            
            # Handle speaker reference
            speaker_path = None
            if speaker_ref:
                speaker_path = os.path.join(app.config['UPLOAD_FOLDER'], 'speaker_ref.wav')
                speaker_ref.save(speaker_path)
            
            # Generate final audio
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.wav')
            tts.generate_audio(translations, output_path, speaker_path)
            
            return send_file(output_path, as_attachment=True)
        
        except Exception as e:
            return f"Error processing request: {str(e)}", 500
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)