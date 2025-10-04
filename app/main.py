import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel,EmailStr
from vector_store import add_document, search, embed_text, get_user_collection
from cerebras.cloud.sdk import Cerebras
from PyPDF2 import PdfReader
import mysql.connector
import random
import string

app = FastAPI(title="Cerebras RAG AI Assistant")
client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

class ChatRequest(BaseModel):
    user_id: str
    query: str

# Utility: split text into chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# ----------------------------
# Upload file (user-specific)
# ----------------------------
@app.post("/upload-file")
async def upload_file(
    user_id: str = Form(...), 
    file: UploadFile = File(...)
):
    """
    Upload a file for a specific user, extract text, chunk it, and store embeddings.
    """
    # Ensure storage directory exists
    storage_dir = f"app/file_storage/{user_id}"
    os.makedirs(storage_dir, exist_ok=True)

    # Save file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(storage_dir, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract text
    text = ""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        with open(file_path, "r", errors="ignore") as f:
            text = f.read()

    # Chunk + store (user-specific)
    chunks = chunk_text(text)
    add_document(user_id, file_id, chunks, source=file.filename)

    return {
        "message": "File uploaded and indexed",
        "file_id": file_id,
        "source": file.filename,
        "user_id": user_id
    }

# /*
# curl -X POST "http://127.0.0.1:8000/upload-file" \
#   -F "user_id=user123" \
#   -F "file=@/path/to/example.pdf"
# */



# ----------------------------
# Chat endpoint (user-specific)
# ----------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    """
    Search documents for the user and respond using Cerebras LLM.
    """
    # Search vector DB scoped to user
    results = search(request.user_id, request.query, top_k=3)
    retrieved_docs = results["documents"][0]
    sources = results["metadatas"][0]

    context_text = "\n".join(
        [f"Source: {src['source']}\nContent: {doc}" for doc, src in zip(retrieved_docs, sources)]
    )

    # System prompt
    SYSTEM_PROMPT = """
You are a helpful and knowledgeable assistant. Your responses must follow these rules STRICTLY:

**WHEN THE ANSWER IS IN THE CONTEXT:**
- Provide a clear, specific answer using ONLY the information from the provided context
- Always cite your source by mentioning which document the information came from
- Write in a friendly, casual tone like a knowledgeable person explaining something
- Be concise but thorough - give the complete answer found in the context

**WHEN THE ANSWER IS NOT IN THE CONTEXT:**
- DO NOT try to make up an answer or use outside knowledge
- Politely state that you couldn't find the specific information in the provided documents
- Offer a friendly suggestion for how the user might find the information
- Keep it warm and human-like - don't sound robotic or apologetic

Remember: Your knowledge is limited to exactly what's in the provided context documents.
"""

    # Cerebras chat call
    response = client.chat.completions.create(
        model="llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Here is my question: {request.query}

Here are the documents I have for context:
{context_text}

Please answer my question using ONLY the information above. If the answer isn't there, just let me know politely."""}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [src['source'] for src in sources],
        "user_id": request.user_id
    }


# {
#   "user_id": "user123",
#   "query": "What is mentioned about the first chunk?"
# }
# curl -X POST "http://127.0.0.1:8000/chat" \
#   -H "Content-Type: application/json" \
#   -d '{"user_id": "user123", "query": "What is mentioned about the first chunk?"}'




# ----------------------------
# Database Connection
# ----------------------------
def get_db():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
        database=os.getenv("MYSQL_DB", "authdb")
    )

# ----------------------------
# Auth Models
# ----------------------------
class OTPRequest(BaseModel):
    email: EmailStr

class VerifyRequest(BaseModel):
    user_id: str
    otp: str

# ----------------------------
# OTP Helpers
# ----------------------------
def generate_otp(length=6):
    return ''.join(random.choices(string.digits, k=length))

@app.post("/send-otp")
def send_otp(req: OTPRequest):
    conn = get_db()
    cursor = conn.cursor()

    otp = generate_otp()

    # Check if email already exists
    cursor.execute("SELECT id FROM otps WHERE email = %s", (req.email,))
    row = cursor.fetchone()   # ✅ consume result here

    if row:
        user_id = row[0]
        # Clear any pending results just in case
        cursor.fetchall()  # ✅ ensures no unread results
        cursor.execute("UPDATE otps SET otp = %s WHERE id = %s", (otp, user_id))
        message = "OTP updated for existing user"
    else:
        cursor.execute("INSERT INTO otps (email, otp) VALUES (%s, %s)", (req.email, otp))
        user_id = cursor.lastrowid
        message = "OTP generated for new user"

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "status": "success",
        "user_id": str(user_id),
        "email": req.email,
        "otp": otp,
        "message": message
    }


@app.post("/verify-otp")
def verify_otp(req: VerifyRequest):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT otp FROM otps WHERE id=%s ORDER BY created_at DESC LIMIT 1", (req.user_id,))
    row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="OTP not found")

    if row[0] == req.otp:
        return {"status": "success", "message": "OTP verified"}
    else:
        raise HTTPException(status_code=400, detail="Invalid OTP")




# //OLD FILE CODE
# import os
# import uuid
# from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel
# from vector_store import add_document, search, embed_text

# from cerebras.cloud.sdk import Cerebras
# from PyPDF2 import PdfReader

# app = FastAPI(title="Cerebras RAG AI Assistant")
# client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

# class ChatRequest(BaseModel):
#     query: str

# # Utility: split text into chunks
# def chunk_text(text, chunk_size=500):
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# @app.post("/upload-file")
# async def upload_file(file: UploadFile = File(...)):
#     # Ensure storage directory exists
#     storage_dir = "app/file_storage"
#     os.makedirs(storage_dir, exist_ok=True)
#     # Save file
#     file_id = str(uuid.uuid4())
#     file_path = os.path.join(storage_dir, f"{file_id}_{file.filename}")
#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     # Extract text
#     text = ""
#     if file.filename.endswith(".pdf"):
#         reader = PdfReader(file_path)
#         text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
#     else:
#         with open(file_path, "r", errors="ignore") as f:
#             text = f.read()

#     # Chunk + store
#     chunks = chunk_text(text)
#     add_document(file_id, chunks, source=file.filename)
#     print("hgELLO")
#     return {
#         "message": "File uploaded and indexed",
#         "file_id": file_id,
#         "source": file.filename
#     }

# @app.post("/chat")
# def chat(request: ChatRequest):
#     # Search vector DB
#     results = search(request.query, top_k=3)
#     retrieved_docs = results["documents"][0]
#     sources = results["metadatas"][0]

#     context_text = "\n".join(
#         [f"Source: {src['source']}\nContent: {doc}" for doc, src in zip(retrieved_docs, sources)]
#     )

#     # Enhanced system prompt
#     SYSTEM_PROMPT = """
#     You are a helpful and knowledgeable assistant. Your responses must follow these rules STRICTLY:

#     **WHEN THE ANSWER IS IN THE CONTEXT:**
#     - Provide a clear, specific answer using ONLY the information from the provided context
#     - Always cite your source by mentioning which document the information came from
#     - Write in a friendly, casual tone like a knowledgeable person explaining something
#     - Be concise but thorough - give the complete answer found in the context

#     **WHEN THE ANSWER IS NOT IN THE CONTEXT:**
#     - DO NOT try to make up an answer or use outside knowledge
#     - Politely state that you couldn't find the specific information in the provided documents
#     - Offer a friendly suggestion for how the user might find the information
#     - Keep it warm and human-like - don't sound robotic or apologetic

#     **RESPONSE FORMAT EXAMPLES:**
#     Good answer (when info exists):
#     "Based on the project documentation, the API key should be stored in environment variables for security. The specific environment variable name is 'CEREBRAS_API_KEY'."

#     Good non-answer (when info doesn't exist):
#     "Thanks for your question! I've looked through the available documents, but I couldn't find specific information about that topic. You might want to check the official documentation or reach out to the support team for more detailed help."

#     Remember: Your knowledge is limited to exactly what's in the provided context documents.
#     """

#     # In your chat function
#     response = client.chat.completions.create(
#         model="llama-4-scout-17b-16e-instruct",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": f"""Here is my question: {request.query}

#     Here are the documents I have for context:
#     {context_text}

#     Please answer my question using ONLY the information above. If the answer isn't there, just let me know politely."""}
#         ]
#     )

#     return {
#         "answer": response.choices[0].message.content,
#         "sources": [src['source'] for src in sources]
#     }
