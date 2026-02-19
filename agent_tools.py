import random
from typing import List, Dict, Any
from langchain.tools import tool

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
        Simulates filing an insurance claim.
        
        Args:
            user_id (str): The ID of the user filing the claim.
            policy_type (str): The type of policy (e.g., 'Motor', 'Home').
            amount (float): The amount claimed.
            
        Returns:
            str: Status message with claim ID or review requirement.
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
    return db.file_claim(user_id, policy_type, amount)

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
