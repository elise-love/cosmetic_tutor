from openai import OpenAI
import io
import threading
import time
import pyaudio
import wave
import tempfile
from pathlib import Path
from typing import Optional, Callable
import pygame

from cosmetic_tutor.types import BlushMetrics
from cosmetic_tutor.config import OpenAIConfig


class OpenAIAssistant:
    """OpenAI assistant with STT and TTS capabilities for cosmetic tutoring."""

    def __init__(self, config: OpenAIConfig):
        self.config = config

        # ✅ 正確初始化 OpenAI Client（2.x）
        self.client = OpenAI(
            api_key=config.api_key,
            organization=config.organization_id or None,
        )

        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024

        # Initialize audio
        self.audio = pyaudio.PyAudio()

        # Initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.audio_frames = []

        # Callbacks
        self.on_transcription_ready: Optional[Callable[[str], None]] = None
        self.on_response_ready: Optional[Callable[[str], None]] = None
        self.on_tts_ready: Optional[Callable[[], None]] = None

    # =========================
    # Recording
    # =========================

    def start_recording(self) -> None:
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_frames = []
        self.recording_thread = threading.Thread(
            target=self._record_audio, daemon=True
        )
        self.recording_thread.start()
        print("[Voice] Recording started...")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

        print("[Voice] Recording stopped, processing...")
        self._process_recorded_audio()

    def _record_audio(self) -> None:
        stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        try:
            while self.is_recording:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.audio_frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()

    # =========================
    # Speech-to-Text (Whisper)
    # =========================

    def _process_recorded_audio(self) -> None:
        if not self.audio_frames:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wave_file = wave.open(temp_file.name, "wb")
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(
                self.audio.get_sample_size(self.audio_format)
            )
            wave_file.setframerate(self.rate)
            wave_file.writeframes(b"".join(self.audio_frames))
            wave_file.close()

            try:
                with open(temp_file.name, "rb") as audio_file:
                    # Fix: Use the correct OpenAI API for speech-to-text
                    transcription = self.client.audio.transcriptions.create(
                        model=self.config.stt_model,  # e.g. "whisper-1"
                        file=audio_file
                    )

                transcription_text = transcription.text.strip()
                print(f"[STT] {transcription_text}")

                if self.on_transcription_ready:
                    self.on_transcription_ready(transcription_text)

            except Exception as e:
                print(f"[STT] Error: {e}")
            finally:
                Path(temp_file.name).unlink(missing_ok=True)

    # =========================
    # GPT Advice
    # =========================

    def get_advice(
        self,
        user_message: str,
        blush_metrics: Optional[BlushMetrics] = None,
    ) -> None:
        threading.Thread(
            target=self._process_advice,
            args=(user_message, blush_metrics),
            daemon=True,
        ).start()

    def _process_advice(
        self,
        user_message: str,
        blush_metrics: Optional[BlushMetrics],
    ) -> None:
        try:
            system_prompt = self._build_system_prompt(blush_metrics)

            # Fix: Use the correct OpenAI API for chat completion
            response = self.client.chat.completions.create(
                model=self.config.model,  # e.g. "gpt-4o-mini"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            advice_text = response.choices[0].message.content.strip()
            print(f"[GPT] {advice_text}")

            if self.on_response_ready:
                self.on_response_ready(advice_text)

            self._text_to_speech(advice_text)

        except Exception as e:
            print(f"[GPT] Error: {e}")

    # =========================
    # Prompt
    # =========================

    def _build_system_prompt(
        self, blush_metrics: Optional[BlushMetrics]
    ) -> str:
        prompt = (
            "你是一位專業的彩妝教學助理，"
            "請用友善、專業、具體可操作的方式回答。\n"
            "- 使用繁體中文\n"
            "- 回答要簡潔\n"
        )

        if blush_metrics:
            prompt += f"""
目前腮紅分析：
- 左臉頰紅色增量：{blush_metrics.left_delta_a:.2f}
- 右臉頰紅色增量：{blush_metrics.right_delta_a:.2f}
- 不均勻度：{blush_metrics.uneven_score:.2f}
- 左覆蓋率：{blush_metrics.left_coverage:.1%}
- 右覆蓋率：{blush_metrics.right_coverage:.1%}
- 偏移方向：{blush_metrics.offset_dir}
"""

        return prompt

    # =========================
    # Text-to-Speech
    # =========================

    def _text_to_speech(self, text: str) -> None:
        try:
            response = self.client.audio.speech.create(
                model=self.config.tts_model,  # e.g. "tts-1"
                voice=self.config.tts_voice,
                input=text,
            )

            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as temp_file:
                response.stream_to_file(temp_file.name)

                pygame.mixer.music.load(temp_file.name)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                if self.on_tts_ready:
                    self.on_tts_ready()

                Path(temp_file.name).unlink(missing_ok=True)

        except Exception as e:
            print(f"[TTS] Error: {e}")

    # =========================
    # Cleanup
    # =========================

    def cleanup(self) -> None:
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()

        self.audio.terminate()
        pygame.mixer.quit()
