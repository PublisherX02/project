import random
from typing import List, Dict, Any
from langchain.tools import tool
import requests
import json

class InsuranceDatabase:
    """
    Simulated Insurance Database for tracking policies and claims.
    """
    def __init__(self):
        # Hardcoded dummy database
        self.users = {
            "USER123": {"policy_type": "Motor", "status": "Active", "coverage": 50000},
            "USER456": {"policy_type": "Home", "status": "Active", "coverage": 100000},
            "USER789": {"policy_type": "Health", "status": "Expired", "coverage": 0}
        }
        self.claims = {}

    def file_claim(self, user_id: str, policy_type: str, amount: float) -> str:
        """
        Simulates filing an insurance claim internally (Legacy/Fallback).
        Now primarily used by the Secure API, not directly by the tool.
        """
        if user_id not in self.users:
            return f"❌ User {user_id} not found in database."
            
        if self.users[user_id]["policy_type"] != policy_type:
            return f"❌ User {user_id} does not have a {policy_type} policy."

        # claims > 5000 require human review
        if amount > 5000:
            return f"⚠️ Claim amount {amount} exceeds automatic approval limit. Sent for human review. Reference: REV-{random.randint(1000, 9999)}"
        
        # Generate fake claim ID
        claim_id = f"CLM-{random.randint(10000, 99999)}"
        self.claims[claim_id] = {
            "user_id": user_id,
            "amount": amount,
            "status": "Approved"
        }
        return f"✅ Claim filed successfully! Your Claim ID is {claim_id}."

    def check_policy(self, user_id: str) -> str:
        """
        Checks the status of a user's insurance policy.
        
        Args:
            user_id (str): The ID of the user.
            
        Returns:
            str: Policy details or error message.
        """
        user = self.users.get(user_id)
        if not user:
            return f"❌ User {user_id} not found."
        
        return f"📋 Policy Details for {user_id}:\n- Type: {user['policy_type']}\n- Status: {user['status']}\n- Coverage: ${user['coverage']}"

# Instantiate Database
db = InsuranceDatabase()

# Create LangChain Tools
@tool
def file_claim_tool(user_id: str, policy_type: str, amount: float) -> str:
    """
    Use this tool to file an insurance claim for a user.
    Requires user_id (str), policy_type (str), and amount (float).
    Returns the claim status and ID.
    """
    # Define the API endpoint
    api_url = "http://localhost:8000/api/secure_claim"
    
    # Define the headers with the security token
    headers = {
        "Content-Type": "application/json",
        "X-Token": "Imani_Secure_2026"
    }
    
    # Create the payload
    payload = {
        "user_id": user_id,
        "policy_type": policy_type,
        "amount": amount
    }
    
    try:
        # Make the POST request to the secure API
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()
            return f"✅ {response_data['message']} Claim ID: {response_data['claim_id']}"
            
        elif response.status_code == 401:
            return "🚨 Security System Blocked Request: Unauthorized Access (Invalid Token)."
            
        elif response.status_code == 422:
            return f"🚨 Security System Blocked Request: Validation Error. Please check your inputs (Amount limit: 50,000). Details: {response.text}"
            
        else:
            return f"❌ Error filing claim. Server returned status {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to the Secure API Gateway. Is the security_api.py server running?"
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

# Export list of tools
insurance_tools = [file_claim_tool, check_policy_tool]
