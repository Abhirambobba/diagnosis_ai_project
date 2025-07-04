from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from rag_chatbot import handle_prompt_and_pdf
from langchain_helper import learn

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Diagnosis AI Backend running"}

@app.post("/diagnose")
async def diagnose_symptoms(prompt: str = Form(...), file: UploadFile = File(None)):
    pdf_text = await file.read() if file else ""
    result = await handle_prompt_and_pdf(prompt, pdf_text)
    return result

@app.post("/learn")
def manual_learn(prompt: str = Form(...), response: str = Form(...)):
    learn(prompt, response)
    return {"message": "Learned successfully!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
