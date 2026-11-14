"""Groq's OpenAI-compatible endpoint. The free tier caps each model by requests and tokens per minute and per day."""

from llm.providers.openai_compat import OpenAICompatible, _snippet, parse_duration  # noqa: F401  (re-exported)


class GroqProvider(OpenAICompatible):
    name = "groq"
    label = "Groq"
    base_url = "https://api.groq.com/openai/v1"
