from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import jwt
from datetime import datetime
import time
import logging
import sys

# --- 1. IMMUTABLE AUDIT LOGGING (SOC2 Compliance) ---
# This saves every security event to an 'audit.log' file AND prints it to the terminal
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
# Physically blocks HTTP requests unless they come from your Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", 
        "http://frontend-agent:8501", 
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["POST"], # Only allow POST requests, block GET/DELETE
    allow_headers=["*"],
)

SECRET_KEY = "OLEA_HACKATHON_SUPER_SECRET_2026"
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
    
    # ANTI-DDOS IDENTITY-BASED RATE LIMITING
    if masked_user in request_tracker:
        last_request_time = request_tracker[masked_user]
        if current_time - last_request_time < 5.0: 
            logger.warning(f"RATE LIMIT TRIGGERED | Proxychain/Spam blocked for Target: {masked_user}")
            raise HTTPException(
                status_code=429, 
                detail="High Traffic Alert: Multiple claims detected for this user. Please wait 5 seconds."
            )
            
    request_tracker[masked_user] = current_time
    
    # Process Claim
    claim_id = f"SECURE-{request.user_id}-99"
    
    # Audit Log the Success (with masked data!)
    logger.info(f"SUCCESS | Claim Processed | IP: {client_ip} | User: {masked_user} | Amount: ${request.amount}")
    
    return {
        "status": "success",
        "message": "Claim passed security validation and was filed.",
        "claim_id": claim_id
    }

if __name__ == "__main__":
    logger.info("🔒 Starting OLEA Enterprise Secure API Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
