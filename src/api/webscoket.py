
from fastapi import APIRouter, WebSocket, WebSocketDisconnect,Request
from fastapi.encoders import jsonable_encoder
from pydantic import  ValidationError



from src.query import retiver_query
from src.schema import ChatRequest
from config import logger

router = APIRouter()



@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    await websocket.accept()

    await websocket.send_json({
        "type": "connected",
        "message": "Connected to RAG chatbot"
    })

    # Get shared resources initialized during FastAPI lifespan
    index = websocket.app.state.index
    llm = websocket.app.state.llm
    try:
        while True:

            # 1. Receive query from frontend
            data = await websocket.receive_json()

            # 2. Validate request
            try:
                chat_request = ChatRequest.model_validate(data)

            except ValidationError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid request. Please provide a valid message."
                })
                continue

            try:
                # 3. Notify frontend that processing has started
                await websocket.send_json({
                    "type": "start",
                    "message": "Generating response..."
                })

                # Callback for every generated token
                async def send_token(token: str):

                    await websocket.send_json({
                        "type": "token",
                        "content": token
                    })

                # 4.Execute streaming RAG pipeline
                response = await retiver_query(
                    query=chat_request.message,
                    index=index,
                    llm=llm,
                    on_token=send_token
                )

                # 5. Send complete LLM response
                await websocket.send_json({
                    "type": "result",
                    "data": jsonable_encoder(response)
                })

                # 6. Notify frontend that processing is complete
                await websocket.send_json({
                    "type": "done"
                })

            except Exception:
                logger.exception("RAG query failed")

                await websocket.send_json({
                    "type": "error",
                    "message": "Unable to process your query"
                })

    except Exception as e:
        logger.error(e)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
