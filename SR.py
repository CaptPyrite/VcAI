import pyaudio
import wave
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import time

def transcribe_speech():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5

    audio_path = "recorded_audio.wav"

    audio_data = []

    def callback(in_data, frame_count, time_info, status):
        audio_data.append(in_data)
        return (in_data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    print("Recording started...")
    stream.start_stream()

    time.sleep(RECORD_SECONDS)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Recording stopped.")

    audio_data = b''.join(audio_data)

    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    audio, sample_rate = sf.read(audio_path)

    inputs = tokenizer(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    input_values = inputs.input_values
    attention_mask = inputs.attention_mask

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return str(transcription).lower()
