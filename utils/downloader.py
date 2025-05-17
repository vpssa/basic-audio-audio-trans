import yt_dlp
import os
import whisper

def download_and_transcribe(video_url, output_dir="transcripts"):
    os.makedirs(output_dir, exist_ok=True)
    transcript_path = os.path.join(output_dir, "transcript_raw.txt")
    
    # Try to download captions
    subtitle_file = _download_captions(video_url, output_dir)
    
    if subtitle_file:
        _convert_subtitles(subtitle_file, transcript_path)
        return transcript_path
    
    # Fallback to Whisper transcription
    audio_path = _download_audio(video_url, output_dir)
    if audio_path:
        return _transcribe_audio(audio_path, transcript_path)
    
    raise Exception("Failed to obtain transcript")

def _download_captions(video_url, output_dir):
    ydl_opts = {
        'writesubtitles': True,
        'subtitleslangs': ['en', 'hi'],
        'skip_download': True,
        'outtmpl': os.path.join(output_dir, 'captions'),
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            for lang_code, sub_info in info_dict.get('requested_subtitles', {}).items():
                if sub_info.get('filepath') and os.path.exists(sub_info['filepath']):
                    return sub_info['filepath']
    except Exception as e:
        print(f"Caption download error: {e}")
    return None

def _convert_subtitles(subtitle_file, output_path):
    with open(subtitle_file, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if line and not line.isdigit() and "-->" not in line and not line.startswith(("WEBVTT", "Kind:", "Language:")):
                f_out.write(line + "\n")

def _download_audio(video_url, output_dir):
    audio_path = os.path.join(output_dir, "downloaded_audio.mp3")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'outtmpl': os.path.join(output_dir, 'audio'),
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        for f in os.listdir(output_dir):
            if f.endswith(".mp3"):
                os.rename(os.path.join(output_dir, f), audio_path)
                return audio_path
    except Exception as e:
        print(f"Audio download error: {e}")
    return None

def _transcribe_audio(audio_path, output_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, fp16=False)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    os.remove(audio_path)
    return output_path