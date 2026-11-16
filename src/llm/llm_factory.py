
from langchain.chat_models import init_chat_model


def create_llm(
    provider: str,
    model_name: str,
    **model_options
):
    """
    Create a chat model independently of its provider.

    Examples:
        create_llm("ollama", "qwen3:4b", base_url="http://localhost:11434")
        create_llm("openai", "gpt-4.1-mini", api_key="...")
    """

    return init_chat_model(
        model=model_name,
        model_provider=provider,
        **model_options
    )
