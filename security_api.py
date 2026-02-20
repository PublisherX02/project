from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field, validator
import uvicorn
import jwt
from datetime import datetime
import time

app = FastAPI(title="OLEA Secure - Enterprise API Gateway")

# --- Enterprise Security Config ---
SECRET_KEY = "OLEA_HACKATHON_SUPER_SECRET_2026"
request_tracker = {}  # Memory-based rate limiter

# --- Security Dependency (JWT) ---
def verify_token(x_token: str = Header(...)):
    """Validates dynamic, expiring JWT tokens."""
    try:
        # Decode the token
        payload = jwt.decode(x_token, SECRET_KEY, algorithms=["HS256"])
        
        # Check if the token has expired
        if payload["exp"] < datetime.utcnow().timestamp():
            raise HTTPException(status_code=401, detail="Security Alert: Token Expired. Possible Replay Attack blocked.")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Security Alert: Token Expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Security Alert: Invalid Token Signature.")

# --- Data Validation Models ---
class ClaimRequest(BaseModel):
    user_id: str = Field(..., min_length=5, description="Unique User ID (Alphanumeric)")
    policy_type: str = Field(..., description="Type of insurance policy (e.g., Motor, Home)")
    amount: float = Field(..., gt=0, le=50000, description="Claim amount (Max 50,000)")

    @validator("user_id", "policy_type")
    def block_sql_injection(cls, value):
        dangerous_keywords = ["SELECT", "DROP", "INSERT", "DELETE", "UPDATE", "UNION", "--", ";"]
        val_upper = value.upper()
        for kw in dangerous_keywords:
            if kw in val_upper:
                raise ValueError("Security Alert: Malicious SQL patterns detected.")
        return value

# --- Endpoints ---
@app.post("/api/secure_claim", dependencies=[Depends(verify_token)])
async def submit_secure_claim(request: ClaimRequest, raw_request: Request):
    """Secure endpoint with Rate Limiting and JWT."""
    
    # 1. ANTI-DDOS RATE LIMITING
    client_ip = raw_request.client.host
    current_time = time.time()
    
    if client_ip in request_tracker:
        last_request_time = request_tracker[client_ip]
        # Limit: 1 request per 10 seconds per IP
        if current_time - last_request_time < 10.0: 
            raise HTTPException(
                status_code=429, 
                detail="High Traffic Alert: Please wait before submitting another claim. You are in the queue."
            )
            
    # Update the tracker
    request_tracker[client_ip] = current_time
    
    # 2. Process Claim
    claim_id = f"SECURE-{request.user_id}-99"
    
    return {
        "status": "success",
        "message": "Claim passed security validation and was filed.",
        "claim_id": claim_id
    }

if __name__ == "__main__":
    print("🔒 Starting OLEA Enterprise Secure API Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
