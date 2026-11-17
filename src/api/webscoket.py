
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from src.query import retiver_query
from src.schema import ChatRequest
from src.config import logger

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    await websocket.accept()

    await websocket.send_json({
        "type": "connected",
        "message": "Connected to SupportDesk"
    })

    index = websocket.app.state.index
    llm = websocket.app.state.llm

    try:
        while True:
            data = await websocket.receive_json()

            try:
                chat_request = ChatRequest.model_validate(data)
            except ValidationError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid request. Please provide a valid message."
                })
                continue

            try:
                await websocket.send_json({
                    "type": "start",
                    "message": "Retrieving policy evidence..."
                })

                async def send_token(token: str):
                    await websocket.send_json({
                        "type": "token",
                        "content": token
                    })

                async def send_evidence(evidence: list):
                    await websocket.send_json({
                        "type": "evidence",
                        "data": jsonable_encoder(evidence)
                    })
                    await websocket.send_json({
                        "type": "status",
                        "message": "Writing grounded answer..."
                    })

                payload = await retiver_query(
                    query=chat_request.message,
                    index=index,
                    llm=llm,
                    on_token=send_token,
                    on_evidence=send_evidence,
                )

                decision = payload["decision"]
                timings = payload["timings"]

                await websocket.send_json({
                    "type": "result",
                    "data": {
                        "status": decision.status,
                        "message": decision.message,
                        "evidence_id": decision.evidence_id,
                        "timings": timings,
                    }
                })

                await websocket.send_json({
                    "type": "done"
                })

            except Exception:
                logger.exception("RAG query failed")
                await websocket.send_json({
                    "type": "error",
                    "message": "Unable to process your query"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
