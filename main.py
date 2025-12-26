import os
import json
import uvicorn
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from io import BytesIO
from pypdf import PdfReader

# --- 1. SETUP & CONFIGURATION ---
app = FastAPI()

# Allow CORS so Replit's webview works fine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REPLIT SECRET HANDLING
# We try to get the key from Replit Secrets (Environment Variables)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not found in Secrets!")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini Model
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Global context storage (RAM)
current_analysis_context = {}

class ChatRequest(BaseModel):
    message: str

# --- 2. HELPER FUNCTIONS ---

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Reads Excel/CSV/PDF and returns string data."""
    if filename.endswith('.csv'):
        df = pd.read_csv(BytesIO(file_content))
        return df.to_csv(index=False)
    elif filename.endswith(('.xls', '.xlsx')):
        xls = pd.ExcelFile(BytesIO(file_content))
        text_data = ""
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            text_data += f"\n--- Sheet: {sheet} ---\n"
            text_data += df.to_csv(index=False)
        return text_data
    elif filename.endswith('.pdf'):
        reader = PdfReader(BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        raise ValueError("Unsupported file format. Please use CSV, Excel, or PDF.")

# --- 3. PROMPTS ---

SYSTEM_PROMPT_ANALYSIS = """
You are an expert CFO and Financial Analyst AI for 'Ledgrix'.
Your task is to analyze the provided financial data and return a JSON object for a dashboard.

STRICT JSON OUTPUT FORMAT REQUIRED:
{
    "kpi": {
        "cash_available": "Latest cash balance (e.g. '$1.2M')",
        "runway_months": "Calculated runway (number, e.g. 8.5)",
        "runway_date": "Date cash runs out (e.g. 'Jul-26')",
        "monthly_burn": "Avg monthly burn (e.g. '$195K')",
        "status": "Healthy" or "Warning" or "Critical"
    },
    "charts": {
        "runway": {
            "labels": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "data": [1000, 800, 600, 400, 200, 0]
        },
        "burn": {
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "data": [50, 55, 45, 60, 50, 55]
        },
        "expenses": {
            "labels": ["Payroll", "Server", "Marketing", "Office", "Legal"],
            "data": [40, 30, 15, 10, 5]
        }
    },
    "summary": "A 2-sentence executive summary of the financial situation.",
    "raw_data_summary": "Hidden context for the chatbot."
}
"""

# --- 4. API ENDPOINTS ---

@app.get("/")
async def get_app():
    # Serve the HTML file
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>")

@app.post("/api/upload")
async def analyze_file(file: UploadFile = File(...)):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="API Key missing. Set GOOGLE_API_KEY in Secrets.")
        
    try:
        content = await file.read()
        raw_text = extract_text_from_file(content, file.filename)
        
        # Analyze with Gemini
        prompt = f"{SYSTEM_PROMPT_ANALYSIS}\n\nDATA:\n{raw_text}"
        response = model.generate_content(prompt)
        
        # Clean Markdown
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        # Update Context
        global current_analysis_context
        current_analysis_context = {
            "raw_data": raw_text[:20000], # Limit context size
            "analysis": data
        }
        
        return data

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_with_finance(request: ChatRequest):
    if not current_analysis_context:
        return {"response": "Please upload a file first so I have data to analyze!"}

    context_prompt = f"""
    You are a Financial Assistant.
    Context: {current_analysis_context['analysis']['summary']}
    Raw Data Snippet: {current_analysis_context['raw_data'][:5000]}
    
    User: {request.message}
    Answer professionally and concisely.
    """
    
    try:
        response = model.generate_content(context_prompt)
        return {"response": response.text}
    except Exception as e:
        return {"response": "I'm having trouble connecting to the brain right now."}

# --- 5. RUN SERVER ---
if __name__ == "__main__":
    # Replit requires 0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8000)