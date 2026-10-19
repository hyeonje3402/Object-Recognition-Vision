"""
[프론트엔드 연동용 백엔드 서버 - api/server.py]
FastAPI 기반 REST API 및 WebSocket 실시간 이벤트 스트리밍 서버 뼈대
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import List, Dict, Any

app = FastAPI(title="Lab Asset & Lost Property Security API")

# React 프론트엔드 연동을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    """웹소켓 실시간 이벤트 브로드캐스터"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

@app.get("/api/v1/system/status")
async def get_system_status():
    """실습실 보안 시스템 요약 상태 조회 REST API"""
    return {
        "status": "ONLINE",
        "camera_active": True,
        "rois": ["PODIUM", "DOOR", "STUDENT_DESK"],
        "active_rules": ["RULE_1_PUBLIC_ASSET_THEFT", "RULE_2_LOST_PROPERTY"]
    }

@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """
    [실시간 경보 웹소켓]
    - 프론트엔드(React)에서 실시간으로 경보 팝업 및 상태 업데이트를 수신
    - 예: { event: "THEFT_ALERT", asset: "HDMI Cable", timestamp: 1727670000 }
    """
    await manager.connect(websocket)
    try:
        while True:
            # 클라이언트로부터 메시지 수신 대기 (필요 시)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
