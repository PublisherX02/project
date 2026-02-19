# 🌍 Project Imani: The Autonomous, Zero-Trust Insurance Agent for Africa

![Project Status](https://img.shields.io/badge/Status-Hackathon_Ready-success)
![Architecture](https://img.shields.io/badge/Architecture-Zero_Trust-blue)

## 🚀 The Vision
Insurance penetration in Africa is below 3%. Traditional models rely on rigid forms, complex corporate language, and lack trust in digital data security. 

**Imani** is a radical solution: A culturally intelligent, autonomous AI agent that negotiates, processes claims, and speaks the local dialects of the Maghreb, all protected behind a mathematically secure "Zero-Trust" API gateway.

## 🏗️ Architecture & Features

### 1. The "Chameleon" Prompt Engine (Cultural Intelligence)
Imani doesn't just translate; she adapts her persona. The LangChain agent dynamically restructures its prompt based on the user's region, successfully blending French and local Arabic dialects (Arabizi) for:
* 🇹🇳 **Tunisian Arabic (Tounsi)**
* 🇩🇿 **Algerian (Dziri)**
* 🇲🇦 **Moroccan (Darija)**

### 2. Autonomous Action (The "Hands")
Imani is not just a chatbot. Using LangChain `@tool` decorators, she can:
* Query policy statuses in real-time.
* Autonomously extract claim data (User ID, Policy Type, Amount) from natural conversation.
* Route complex actions to secure backend APIs without human intervention.

### 3. The "Zero-Trust" Pipeline (Cybersecurity)
We don't trust the AI blindly. Imani is physically separated from the database by a **FastAPI Guardrail**.
* **Authentication:** The Agent must present a valid `X-Token` header to act.
* **Strict Validation:** Using Pydantic, the API physically rejects hallucinated claims (e.g., amount caps, regex constraints on User IDs) before they touch the database.

### 4. RAG Knowledge Base (The "Brain")
Imani reads messy corporate PDFs and converts them into instant, contextual answers using:
* `meta/llama-3.1-70b-instruct` (Core Reasoning)
* `all-MiniLM-L6-v2` (Embeddings)
* `ChromaDB` (Vector Search)

---

## 💻 How to Run the Demo Locally

You must run both the Security Gateway and the Frontend simultaneously.

### Step 1: Install Requirements
```bash
pip install -r requirements.txt