# app.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import asyncio
import json
import numpy as np

app = FastAPI(title="VisionGuard Real-Time Live Dashboard")

# HTML بسيط للواجهة الحية (Live Stream View)
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>VisionGuard Live SOC Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #38bdf8; }
        .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px; }
        .card { background: #1e293b; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); width: 400px; text-align: left; }
        pre { background: #090d16; padding: 10px; border-radius: 6px; color: #34d399; overflow-x: auto; font-size: 12px; }
        .status { font-weight: bold; color: #f43f5e; }
    </style>
</head>
<body>
    <h1>🛡️ VisionGuard Real-Time Threat Intelligence Dashboard</h1>
    <p>Live WebSocket Stream & Threat Detection Engine</p>
    
    <div class="container">
        <div class="card">
            <h3>🔴 Live Feed Status</h3>
            <p>System Status: <span style="color: #10b981;">Active & Monitoring</span></p>
            <p>Connected Clients: <span id="clients">1</span></p>
            <hr style="border-color: #334155;">
            <h3>🚨 Latest Threat Event</h3>
            <pre id="event-data">Waiting for incoming threat events...</pre>
        </div>
    </div>

    <script>
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            document.getElementById("event-data").innerText = JSON.stringify(data, null, 4);
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected to VisionGuard Live WebSocket Stream!")
    try:
        while True:
            # محاكاة إرسال حدث تهديد لحظي كل ثانيتين للتاكيد والتحقيق
            simulated_event = {
                "system": "VisionGuard Live",
                "status": "Monitoring",
                "alert": "CRITICAL THREAT DETECTED",
                "person_id": 99,
                "behavior": "Fighting & Weapon",
                "confidence": 98.2,
                "severity": "CRITICAL"
            }
            await websocket.send_text(json.dumps(simulated_event))
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        print("🔌 Client disconnected.")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting VisionGuard Live Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)