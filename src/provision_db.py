from config import db_url, db_api_key, COLLECTION_NAME
from src.vector_store import get_client, ensure_collection_exists

def main():

    client = get_client(
        url=db_url,
        api_key=db_api_key,
    )

    try:
        ensure_collection_exists(
            client=client,
            collection=COLLECTION_NAME,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()