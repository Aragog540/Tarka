import os

from anthropic import Anthropic

try:
    from groq import Groq
except ImportError:  # pragma: no cover - handled at runtime if Groq is not installed yet
    Groq = None


def _provider_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError("Set LLM_PROVIDER and the matching API key, or provide GROQ_API_KEY/ANTHROPIC_API_KEY.")


def _default_model(provider: str) -> str:
    if provider == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_text(system_prompt: str, user_prompt: str, *, model: str | None = None, max_tokens: int = 1000, temperature: float = 0.2, json_mode: bool = False) -> str:
    provider = _provider_name()
    model_name = model or _default_model(provider)

    if provider == "groq":
        if Groq is None:
            raise RuntimeError("Groq provider selected but the groq package is not installed.")
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if json_mode else None,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    if provider == "anthropic":
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text.strip()

    raise ValueError(f"Unsupported LLM provider: {provider}")