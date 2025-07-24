import os
from datetime import datetime
import pyaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import wave
import librosa
import tempfile
import threading


class ContinuousTranscriber:
    def __init__(
            self,
            model="openai/whisper-medium",
            slice_duration=5,
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
    ):
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model)
        self.slice_duration = slice_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.is_recording = False

        self.audio = pyaudio.PyAudio()

    def start_transcribe(self):
        stream = self.audio.open(
            rate=self.sample_rate,
            channels=self.channels,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        self.is_recording = True
        accumulation = 0
        while self.is_recording:
            stream.start_stream()
            temp_file = tempfile.NamedTemporaryFile(mode='wb+', delete=False)
            print(f"Recording {self.slice_duration * (accumulation + 1)}s audio slice...")
            frames = []

            # Record for the specified duration
            for _ in range(
                    0, int(self.sample_rate / self.chunk_size * self.slice_duration)
            ):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk_size)
                frames.append(data)

            stream.stop_stream()
            with wave.open(temp_file.name, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(frames))

            print(f'[{datetime.now().strftime("%H:%M:%S")}] Audio slice {accumulation + 1} recorded.')

            if len(frames) > 0:
                threading.Thread(target=self.transcribe_audio, args=(frames,)).start()
            else:
                print("No audio recorded in this slice.")
            accumulation += 1

        stream.stop_stream()
        stream.close()

    def transcribe_audio(self, filename: str):
        """Transcribe the audio file using the model."""
        audio_array, sr = librosa.load(filename, sr=self.sample_rate)
        audio_input = self.processor(
            audio_array, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = self.model.generate(audio_input.float())
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        result = transcription.strip()
        print(result)

    def process_audio_slice(self, slice_number):
        """Process a single audio slice - record and transcribe"""
        temp_filename = f"temp_audio_slice_{slice_number}.wav"

        try:
            # Record audio slice
            if self.start_transcribe(temp_filename):
                # Transcribe the audio
                transcript = self.transcribe_audio(temp_filename)

                # Display results
                timestamp = datetime.now().strftime("%H:%M:%S")
                if transcript:
                    print(f"[{timestamp}] Slice {slice_number}: {transcript}")
                else:
                    print(f"[{timestamp}] Slice {slice_number}: (no speech detected)")

        except Exception as e:
            print(f"Error processing slice {slice_number}: {e}")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def cleanup(self):
        """Clean up resources"""
        self.is_recording = False
        self.audio.terminate()


# Example usage
if __name__ == "__main__":
    # Create recognizer instance
    recognizer = ContinuousTranscriber(
        slice_duration=5,  # 5-second slices
        sample_rate=16000,  # 16kHz sample rate (good for speech)
        channels=1,  # Mono audio
        chunk_size=1024,  # Chunk size for reading audio
    )

    # Start continuous recognition
    file_name = "transcription_output.mp3"
    recognizer.start_transcribe()

    # Clean up
    recognizer.cleanup()
    print("Audio recognition stopped.")
