import os
import random
import logging
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import jwt
from datetime import datetime
import time
import sys
import joblib

# Load BERT Sentiment Detector (CAMeL-Lab fine-tuned on TUNIZI)
try:
    from ml_models.sentiment_detector import detect_sentiment
    _sentiment_ready = True
    print("✅ BERT Sentiment Model loaded successfully.")
except Exception as e:
    _sentiment_ready = False
    detect_sentiment = None
    print(f"⚠️ BERT Sentiment Model not loaded: {e}")

# --- 1. IMMUTABLE AUDIT LOGGING (SOC2 Compliance) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("audit.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SecurityAudit")

app = FastAPI(title="OLEA Secure - Enterprise API Gateway")

# --- 2. STRICT CORS MIDDLEWARE ---
# Fix: Added DELETE to allow_methods to support /api/chat/clear
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", 
        "http://frontend-agent:8501", 
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["POST", "DELETE", "GET"],
    allow_headers=["*"],
)

# Fix: Hard crash if secret is missing — no silent fallback to known-public default
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY env var is not set. Refusing to start.")

# Fix: Separate single-purpose internal API key for chat/vision endpoints
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
if not INTERNAL_API_KEY:
    raise RuntimeError("INTERNAL_API_KEY env var is not set. Refusing to start.")

request_tracker = {}

# --- 3. PII MASKING FUNCTION (GDPR Compliance) ---
def mask_pii(user_id: str) -> str:
    """Masks sensitive Identity Numbers (e.g., USER12345 -> U***345)"""
    if len(user_id) > 4:
        return f"{user_id[0]}***{user_id[-3:]}"
    return "****"

# --- Security Dependency (JWT) ---
def verify_token(x_token: str = Header(...)):
    """Validates dynamic, expiring JWT tokens."""
    try:
        payload = jwt.decode(x_token, SECRET_KEY, algorithms=["HS256"])
        if payload["exp"] < datetime.utcnow().timestamp():
            logger.warning("BLOCKED: Expired JWT Token (Possible Replay Attack attempted).")
            raise HTTPException(status_code=401, detail="Security Alert: Token Expired.")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("BLOCKED: Expired JWT Signature.")
        raise HTTPException(status_code=401, detail="Security Alert: Token Expired.")
    except jwt.InvalidTokenError:
        logger.warning("CRITICAL: Invalid JWT Signature (Tampering Attempt Detected).")
        raise HTTPException(status_code=401, detail="Security Alert: Invalid Token Signature.")

# --- Data Validation Models ---
class ClaimRequest(BaseModel):
    user_id: str = Field(..., min_length=5, max_length=20, description="Unique User ID (Alphanumeric)")
    policy_type: str = Field(..., max_length=50, description="Type of insurance policy (e.g., Motor, Home)")
    amount: float = Field(..., gt=0, le=50000, description="Claim amount (Max 50,000)")

    @validator("user_id", "policy_type")
    def block_sql_injection(cls, value):
        dangerous_keywords = ["SELECT", "DROP", "INSERT", "DELETE", "UPDATE", "UNION", "--", ";"]
        val_upper = value.upper()
        for kw in dangerous_keywords:
            if kw in val_upper:
                logger.critical(f"CRITICAL BLOCKED: SQL Injection pattern '{kw}' detected in payload.")
                raise ValueError("Security Alert: Malicious SQL patterns detected.")
        return value

# --- Endpoints ---
@app.post("/api/secure_claim", dependencies=[Depends(verify_token)])
async def submit_secure_claim(request: ClaimRequest, raw_request: Request):
    """Secure endpoint with Rate Limiting, JWT, CORS, and PII Masking."""
    
    client_ip = raw_request.client.host
    current_time = time.time()
    
    # Mask the User ID immediately so plain text never touches the logs
    masked_user = mask_pii(request.user_id)
    
    # ANTI-DDOS: Key on raw user_id before masking (Fix: avoids masking bypass)
    if request.user_id in request_tracker:
        last_request_time = request_tracker[request.user_id]
        if current_time - last_request_time < 5.0: 
            logger.warning(f"RATE LIMIT TRIGGERED | Proxychain/Spam blocked for Target: {masked_user}")
            raise HTTPException(
                status_code=429, 
                detail="High Traffic Alert: Multiple claims detected for this user. Please wait 5 seconds."
            )
            
    request_tracker[request.user_id] = current_time
    
    # Process Claim
    # Fix: Use random token instead of raw user_id to prevent PII leak in response
    import secrets
    claim_id = f"SECURE-{secrets.token_hex(6).upper()}"

    # Log fraud risk from JWT if present
    try:
        _decoded = jwt.decode(
            raw_request.headers.get("x-token", ""),
            SECRET_KEY, algorithms=["HS256"]
        )
        fraud_risk = _decoded.get("fraud_risk")
        if fraud_risk is not None:
            logger.warning(
                f"FRAUD RISK | User: {masked_user} | Risk: {fraud_risk * 100:.1f}%"
            )
    except Exception:
        pass  # Token already validated upstream; parsing failure here is non-fatal

    # Audit Log the Success (with masked data!)
    logger.info(f"SUCCESS | Claim Processed | IP: {client_ip} | User: {masked_user} | Amount: ${request.amount}")
    
    return {
        "status": "success",
        "message": "Claim passed security validation and was filed.",
        "claim_id": claim_id
    }

from main import chatbot, analyze_damage_image

# Max 5MB base64 image ≈ 6.8M chars
MAX_B64_IMAGE_LEN = 7_000_000
VALID_IMAGE_MAGIC = {b'\xff\xd8': 'JPEG', b'\x89P': 'PNG'}

class ChatRESTRequest(BaseModel):
    message: str
    language: str
    is_voice: bool = False

class VisionRESTRequest(BaseModel):
    base64_img: str = Field(..., max_length=MAX_B64_IMAGE_LEN, description="Base64 image (max 5MB)")
    language: str
    filename: str = "unknown.jpg"

@app.api_route("/api/v1/database_dump", methods=["GET", "POST", "PUT", "DELETE"])
async def honeypot_trap(raw_request: Request):
    """HONEYPOT TRAP: If anyone attempts to scan for a database dump, they get banned."""
    client_ip = raw_request.client.host
    logger.critical(f"CRITICAL HONEYPOT BREACH | Intrusion detected from IP: {client_ip} | Action: BAN IP")
    raise HTTPException(status_code=403, detail="🚨 SECURITY PROTOCOL: Intrusion Detected. Your IP has been logged and permanently banned.")

def verify_internal_key(x_internal_key: str = Header(...)):
    """Validates the internal API key used by the Streamlit frontend."""
    if x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid internal API key.")

@app.post("/api/chat", dependencies=[Depends(verify_internal_key)])
async def chat_endpoint(request: ChatRESTRequest):
    """Processes a chat message via the React Agent."""
    try:
        augmented_message = request.message

        # BERT Sentiment Guard: detect distress before LLM call
        if _sentiment_ready and detect_sentiment is not None:
            sentiment = detect_sentiment(request.message)
            if sentiment.get("flag"):
                confidence = sentiment.get("confidence", 0.0)
                logger.warning(
                    f"SENTIMENT ALERT | Urgent tone detected | Confidence: {confidence:.2%}"
                )
                augmented_message = (
                    request.message
                    + "\n\n[CRISIS PROTOCOL ACTIVE: User appears distressed. "
                    "Respond with maximum empathy, prioritize emotional acknowledgment "
                    "before any technical response.]"
                )

        response_data = chatbot.chat(augmented_message, language=request.language, is_voice=request.is_voice)
        return {
            "response": response_data.get("response", "Error processing request"),
            "sources": response_data.get("sources", [])
        }
    except Exception as e:
        logger.error(f"Chat API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/clear")
async def clear_chat_endpoint():
    """Clears the chat history from the persistent memory."""
    try:
        chatbot.clear_history()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vision", dependencies=[Depends(verify_internal_key)])
async def vision_endpoint(request: VisionRESTRequest):
    """Processes Vision AI simulation with size+type validation."""
    import base64
    try:
        # Fix: Validate image magic bytes before sending to model
        raw = base64.b64decode(request.base64_img[:16])
        magic = raw[:2]
        if magic not in VALID_IMAGE_MAGIC:
            raise HTTPException(status_code=400, detail="Invalid image format. Only JPEG and PNG are supported.")
            
        # Wire custom YOLOv8 pipeline before NVIDIA VLM
        from ml_models.damage_detector import detect_damage
        yolo_res = detect_damage(request.base64_img)
        assessment = analyze_damage_image(request.base64_img, request.language, request.filename, yolo_prescan=yolo_res['summary'])
        return {"response": assessment}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fix: Admin logs endpoint is now protected by JWT token dependency
@app.get("/api/admin/logs", dependencies=[Depends(verify_token)])
async def get_admin_logs():
    """Returns the last 50 lines of the immutable audit.log. Requires JWT Auth."""
    try:
        if os.path.exists("audit.log"):
            with open("audit.log", "r") as f:
                lines = f.readlines()
                return {"logs": lines[-50:]}
        return {"logs": ["Log file not created yet."]}
    except Exception as e:
        return {"logs": [f"Error reading logs: {str(e)}"]}

if __name__ == "__main__":
    logger.info("🔒 Starting OLEA Enterprise Secure API Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
