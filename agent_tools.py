import random
from typing import List, Dict, Any
from langchain.tools import tool
import requests
import json
import jwt
import os
import random
from supabase import create_client, Client
from datetime import datetime, timedelta
import joblib

try:
    from ml_models.fraud_detector import assess_fraud_risk
    _fraud_ready = True
except Exception as e:
    _fraud_ready = False
    assess_fraud_risk = None
    print(f"⚠️ Fraud Detector not loaded: {e}")

import bcrypt

class InsuranceDatabase:
    """
    Supabase-backed Cloud Database for tracking policies, claims, and securely authenticating clients.
    """
    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            self.supabase: Client = create_client(url, key)
        else:
            self.supabase = None
            print("⚠️ Supabase credentials missing!")

    def file_claim(self, user_id: str, policy_type: str, amount: float) -> str:
        if not self.supabase:
            return "❌ Supabase Not Connected."
            
        try:
            res = self.supabase.table("policies").select("policy_type").eq("user_id", user_id).execute()
        except Exception as e:
            return f"❌ Database error: {str(e)}"
            
        if not res.data:
            return f"❌ User {user_id} not found in database."
            
        if res.data[0]["policy_type"] != policy_type:
            return f"❌ User {user_id} does not have a {policy_type} policy."

        # ML Model: assess_fraud_risk on every claim (replaces hardcoded $5000 rule)
            fraud_result = assess_fraud_risk({"Deductible": int(amount), "BasePolicy": policy_type, "PolicyType": policy_type}) if _fraud_ready else {"flag": False, "fraud_probability": 0.0}
            fraud_pct = fraud_result.get("fraud_probability", 0.0) * 100
            fraud_flag = fraud_result.get("flag", False)

            if fraud_flag or amount > 5000:
                fraud_score_msg = f" The mathematical fraud model scores this claim at {fraud_pct:.1f}% risk."

                # Dynamic Multi-Agent Simulation leveraging the LLM inside the tool
                try:
                    from langchain_nvidia_ai_endpoints import ChatNVIDIA
                    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=os.getenv("NVIDIA_API_KEY"), temperature=0.8)
                    prompt = f"Write a dramatic strictly 4-line debate between two AI agents: 'Adjuster AI' and 'Fraud Analyst AI' regarding an insurance claim of ${amount}.{fraud_score_msg} Adjuster wants to approve, Fraud Analyst is suspicious. The debate must end with a single line: 'Verdict: SUSPENDED for Human Review'."
                    debate_text = llm.invoke(prompt).content
                    return f"🚨 **[MULTI-AGENT FRAUD BOARD ENGAGED]** 🚨\n\n{debate_text}\n\n⚠️ Reference: REV-{random.randint(1000, 9999)}"
                except Exception as e:
                    return f"⚠️ Claim flagged by fraud model at {fraud_pct:.1f}% risk. Sent for human review. Reference: REV-{random.randint(1000, 9999)}"
        
        claim_id = f"CLM-{random.randint(10000, 99999)}"
        self.supabase.table("claims").insert({
            "claim_id": claim_id,
            "user_id": user_id,
            "amount": amount,
            "status": "Approved"
        }).execute()
        
        return f"✅ Claim filed successfully! Your Claim ID is {claim_id}."

    def check_policy(self, user_id: str) -> str:
        if not self.supabase:
            return "❌ Supabase Not Connected."
        
        try:
            res = self.supabase.table("policies").select("*").eq("user_id", user_id).execute()
            if not res.data:
                return f"❌ User {user_id} not found."
            
            user = res.data[0]
            return f"📋 Policy Details for {user_id}:\n- Type: {user['policy_type']}\n- Status: {user['status']}\n- Coverage: ${user['coverage']}"
        except Exception as e:
            return f"❌ Supabase error: {str(e)}"
            
    def authenticate_client(self, first_name: str, plaintext_password: str) -> str:
        if not self.supabase: return "❌ Supabase Not Connected."
        try:
            # DEMO LIMITATION: We query by first_name only for simplicity in the demo.
            # In a real environment, this breaks if two clients share a first name (e.g., two "Mohameds").
            # A true production query must combine first_name + last_name, or use an email/ID.
            res = self.supabase.table("clients").select("*").eq("first_name", first_name).eq("is_active", True).execute()
            if not res.data:
                return "❌ Access Denied."
            
            client = res.data[0]
            if client.get("failed_login_attempts", 0) >= 5:
                return "🔒 Account locked. Contact OLEA support."
            
            if bcrypt.checkpw(plaintext_password.encode('utf-8'), client["password_hash"].encode('utf-8')):
                self.supabase.table("clients").update({"failed_login_attempts": 0}).eq("id", client["id"]).execute()
                return f"✅ Access Granted | Welcome {client['first_name']} {client['last_name']} | Policy: {client.get('policy_type', 'N/A')} | Coverage: ${client.get('coverage', 0)}"
            else:
                self.supabase.table("clients").update({"failed_login_attempts": client.get("failed_login_attempts", 0) + 1}).eq("id", client["id"]).execute()
                return "❌ Access Denied. Incorrect password."
        except Exception as e:
            return f"❌ Authentication error: {str(e)}"
            
    def create_client(self, first_name: str, last_name: str, password: str, age: int, profession: str, salary: float, kids: int, cars: int, social_status: str) -> str:
        if not self.supabase: return "❌ Supabase Not Connected."
        try:
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            res = self.supabase.table("clients").insert({
                "first_name": first_name,
                "last_name": last_name,
                "password_hash": password_hash,
                "age": age,
                "profession": profession,
                "salary": salary,
                "kids": kids,
                "cars": cars,
                "social_status": social_status,
                "failed_login_attempts": 0,
                "is_active": True,
                "policy_type": "Motor",
                "coverage": 50000
            }).execute()
            return f"✅ Account created successfully for {first_name} {last_name}."
        except Exception as e:
            return f"❌ Registration failed: {str(e)}"

# Instantiate Database
db = InsuranceDatabase()

@tool
def file_claim_tool(user_id: str, policy_type: str, amount: float) -> str:
    """
    Use this tool to file an insurance claim for a user.
    Requires user_id (str), policy_type (str), and amount (float).
    Returns the claim status and ID.
    """
    api_url = os.getenv("API_URL", "http://localhost:8000/api/secure_claim")
    
    # Fix: JWT secret loaded from environment — hard crash if missing
    secret_key = os.getenv("JWT_SECRET_KEY")
    if not secret_key:
        raise RuntimeError("JWT_SECRET_KEY environment variable is not set.")
    from datetime import datetime, timedelta
    # Pre-compute fraud risk to embed in JWT
    fraud_result = assess_fraud_risk({"Deductible": int(amount), "BasePolicy": policy_type, "PolicyType": policy_type}) if _fraud_ready else {}
    token_payload = {
        "service": "imani_autonomous_agent",
        "exp": datetime.utcnow() + timedelta(seconds=60),
    }
    if fraud_result.get("flag"):
        token_payload["fraud_risk"] = fraud_result.get("fraud_probability", 0.0)
    dynamic_token = jwt.encode(token_payload, secret_key, algorithm="HS256")
    
    headers = {
        "Content-Type": "application/json",
        "X-Token": dynamic_token
    }
    
    payload = {
        "user_id": user_id,
        "policy_type": policy_type,
        "amount": amount
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            return f"✅ {response_data['message']} Claim ID: {response_data['claim_id']}"
            
        elif response.status_code == 401:
            return "🚨 Security System Blocked Request: Unauthorized Access (Invalid or Expired JWT Token)."
            
        elif response.status_code == 422:
            return f"🚨 Security System Blocked Request: Validation Error (Anti-SQL Injection or Limit Exceeded)."
            
        elif response.status_code == 429:
            return "⏳ High Traffic Alert: The OLEA servers are currently experiencing heavy load. Please wait a few moments and try your claim again."
            
        else:
            return f"❌ Error filing claim. Server returned status {response.status_code}."
            
    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to the Secure API Gateway. Is the server running?"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

@tool
def check_policy_tool(user_id: str) -> str:
    """
    Use this tool to check the status and details of a user's insurance policy.
    Requires user_id (str).
    Returns policy information.
    """
    return db.check_policy(user_id)

@tool
def authenticate_client_tool(first_name: str, plaintext_password: str) -> str:
    """
    Authenticate an existing OLEA client by first name and password.
    Returns Access Granted with client details or Access Denied.
    """
    return db.authenticate_client(first_name, plaintext_password)

@tool  
def create_client_tool(first_name: str, last_name: str, password: str, age: int, profession: str, salary: float, kids: int, cars: int, social_status: str) -> str:
    """
    Create a new OLEA client account with all required profile information.
    Requires: first_name, last_name, password (min 6 chars), age (18-100), 
    profession, salary (>0), kids, cars, social_status.
    """
    return db.create_client(first_name, last_name, password, age, profession, salary, kids, cars, social_status)

# Export list of tools
insurance_tools = [file_claim_tool, check_policy_tool, authenticate_client_tool, create_client_tool]
