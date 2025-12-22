import time
from typing import Optional, Callable
from cosmetic_tutor.ai.openai_assistant import OpenAIAssistant
from cosmetic_tutor.types import BlushMetrics
from cosmetic_tutor.config import OpenAIConfig

class VoiceInteractionManager:
    """Manages voice interactions for the cosmetic tutor."""
    
    def __init__(self, openai_config: OpenAIConfig):
        self.assistant = OpenAIAssistant(openai_config)
        
        # State management
        self.is_talk_mode_active = False
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        
        # Current metrics for context
        self.current_metrics: Optional[BlushMetrics] = None
        
        # UI callbacks
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response: Optional[Callable[[str], None]] = None
        
        # Set up assistant callbacks
        self.assistant.on_transcription_ready = self._on_transcription_ready
        self.assistant.on_response_ready = self._on_response_ready  
        self.assistant.on_tts_ready = self._on_tts_ready
        
        # Auto-stop recording timer
        self.recording_start_time = 0
        self.max_recording_duration = 10.0  # seconds
        
    def activate_talk_mode(self) -> None:
        """Activate talk mode - start listening for voice input."""
        if self.is_talk_mode_active:
            return
            
        self.is_talk_mode_active = True
        self._start_listening()
        
    def deactivate_talk_mode(self) -> None:
        """Deactivate talk mode."""
        if not self.is_talk_mode_active:
            return
            
        self.is_talk_mode_active = False
        self._stop_listening()
        self._update_status("語音模式已關閉")
        
    def toggle_talk_mode(self) -> bool:
        """Toggle talk mode on/off. Returns new state."""
        if self.is_talk_mode_active:
            self.deactivate_talk_mode()
            return False
        else:
            self.activate_talk_mode()
            return True
            
    def update_metrics(self, metrics: BlushMetrics) -> None:
        """Update current blush metrics for context."""
        self.current_metrics = metrics
        
    def _start_listening(self) -> None:
        """Start listening for voice input."""
        if self.is_listening or self.is_processing or self.is_speaking:
            return
            
        self.is_listening = True
        self.recording_start_time = time.time()
        self.assistant.start_recording()
        self._update_status("🎤 聆聽中... (說完話請按空格鍵停止)")
        
    def _stop_listening(self) -> None:
        """Stop listening and process input."""
        if not self.is_listening:
            return
            
        self.is_listening = False
        self.assistant.stop_recording()
        self.is_processing = True
        self._update_status("🔄 處理中...")
        
    def _on_transcription_ready(self, transcription: str) -> None:
        """Called when STT transcription is ready."""
        if self.on_transcription:
            self.on_transcription(transcription)
            
        if transcription.strip():
            # Get advice from GPT
            self.assistant.get_advice(transcription, self.current_metrics)
        else:
            self.is_processing = False
            if self.is_talk_mode_active:
                self._start_listening()  # Start listening again
            else:
                self._update_status("準備就緒")
                
    def _on_response_ready(self, response: str) -> None:
        """Called when GPT response is ready."""
        if self.on_response:
            self.on_response(response)
            
        self.is_processing = False
        self.is_speaking = True
        self._update_status("🔊 播放回應中...")
        
    def _on_tts_ready(self) -> None:
        """Called when TTS playback is finished."""
        self.is_speaking = False
        
        if self.is_talk_mode_active:
            # Auto-restart listening after speaking
            time.sleep(0.5)  # Brief pause
            self._start_listening()
        else:
            self._update_status("準備就緒")
            
    def _update_status(self, status: str) -> None:
        """Update status message."""
        print(f"[Voice] {status}")
        if self.on_status_change:
            self.on_status_change(status)
            
    def handle_key_press(self, key: int) -> bool:
        """Handle key press events. Returns True if key was handled."""
        # Space bar to stop recording
        if key == ord(' '):
            if self.is_listening:
                self._stop_listening()
                return True
                
        # 't' key to toggle talk mode
        elif key == ord('t'):
            self.toggle_talk_mode()
            return True
            
        return False
        
    def update(self) -> None:
        """Update voice manager state - call this in main loop."""
        # Auto-stop recording after max duration
        if (self.is_listening and 
            time.time() - self.recording_start_time > self.max_recording_duration):
            self._stop_listening()
            
    def cleanup(self) -> None:
        """Clean up resources."""
        self.deactivate_talk_mode()
        self.assistant.cleanup()