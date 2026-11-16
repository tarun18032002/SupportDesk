import asyncio
from ollama import AsyncClient

async def main():

    client = AsyncClient(
        host="http://127.0.0.1:11434",
        timeout=600.0
    )

    print("Sending request to Qwen3...")

    response = await client.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": "Explain RAG in two sentences."
            }
        ],
        stream=False
    )

    print("\nMODEL RESPONSE:")
    print(response.message.content)

    await client._client.aclose()

asyncio.run(main())