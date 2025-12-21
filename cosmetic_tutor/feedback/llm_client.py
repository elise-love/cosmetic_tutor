#llm_client.py
from cosmetic_tutor.types import Advice

class LLMClient:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def generate (self,prompt:str) -> Advice:
        if not self.enabled:
            return Advice(text="(LLM 未啟用)", tags=("fallback",))
        #TODO: integrate API later
        raise NotImplementedError("Connect your LLM provider here.")