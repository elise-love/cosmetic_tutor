#config.py
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass 
class BlushThresholds:
    #starting values
    strong_delta_a: float = 4.0 #delta threshold for "too strong"
    uneven_delta_a: float = 3.0 #abs(left_delta- right_delta)
    coverage_ratio: float = 0.8 #pixels above threshold / ROI area
    red_a_threshold: float = 135.0 #Lab a* threshold (0~255 in OpenCV lab)

    @classmethod
    def from_env(cls) -> 'BlushThresholds':
        """Create BlushThresholds from environment variables."""
        return cls(
            strong_delta_a=float(os.getenv('STRONG_DELTA_A', '4.0')),
            uneven_delta_a=float(os.getenv('UNEVEN_DELTA_A', '3.0')),
            coverage_ratio=float(os.getenv('COVERAGE_RATIO', '0.8')),
            red_a_threshold=float(os.getenv('RED_A_THRESHOLD', '135.0'))
        )

@dataclass
class OpenAIConfig:
    api_key: str
    organization_id: Optional[str] = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 200
    temperature: float = 0.7
    tts_model: str = "tts-1"
    tts_voice: str = "nova"  # Available voices: alloy, echo, fable, onyx, nova, shimmer
    stt_model: str = "whisper-1"

    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        """Create OpenAIConfig from environment variables."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        return cls(
            api_key=api_key,
            organization_id=os.getenv('OPENAI_ORGANIZATION_ID'),
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '200')),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
            tts_model=os.getenv('OPENAI_TTS_MODEL', 'tts-1'),
            tts_voice=os.getenv('OPENAI_TTS_VOICE', 'nova'),
            stt_model=os.getenv('OPENAI_STT_MODEL', 'whisper-1')
        )

@dataclass
class AppConfig:
    #index of the camera used by OpenCV
    #0 refers to default camera
    camera_index: int = 0

    #target frames per second for video capture
    #control performance vs quality
    target_fps: int = 30

    thresholds: BlushThresholds = field(default_factory=BlushThresholds)
    openai: OpenAIConfig = field(default_factory=lambda: OpenAIConfig.from_env())

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create AppConfig from environment variables."""
        return cls(
            camera_index=int(os.getenv('CAMERA_INDEX', '0')),
            target_fps=int(os.getenv('TARGET_FPS', '30')),
            thresholds=BlushThresholds.from_env(),
            openai=OpenAIConfig.from_env()
        )
