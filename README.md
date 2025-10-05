# ⚙️ How to Run the RAG AI Assistant Project

This guide explains how to set up, configure, and run the **RAG AI Assistant** locally or in Docker.

---
## App
https://github.com/vickychhetri/ha_ai_app

## Demo
https://www.youtube.com/watch?v=FQTvGXs-FTs

## 🧩 1. Prerequisites

Before starting, ensure you have the following installed:

- 🐍 **Python 3.11+**
- 🐳 **Docker & Docker Compose** (optional but recommended)
- 🧠 **Cerebras API key** (for accessing the LLaMA model)
- 📦 **Git** (for cloning the repository)

Database 
CREATE DATABASE IF NOT EXISTS `authdb`;
USE `authdb`;
CREATE TABLE IF NOT EXISTS `otps` (
  `id` int NOT NULL AUTO_INCREMENT,
  `email` varchar(255) NOT NULL,
  `otp` varchar(10) NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
);



## 📁 2. Clone the Repository

git clone url
cd folder


docker-compose up --build



# 🧠 RAG AI Assistant with User-wise Contextual Knowledge  
> *“This assistant empowers me to instantly access my personal knowledge anytime, anywhere.”*

---

## 🚀 Project Overview  

**RAG AI Assistant** is a personalized retrieval-augmented generation (RAG) system that allows users to:  
- Login securely  
- Upload documents (PDF/Text)  
- Ask natural language questions  
- Retrieve accurate, context-based answers — powered by **Meta LLaMA** through **Cerebras SDK**  

Each user has an **isolated vector database**, ensuring data privacy and contextual precision.  

---

## 🔑 Key Features  

- 🔐 **User Authentication & Isolation:** Each user gets a unique ID and private vector DB  
- 📄 **Document Upload & Chunking:** Upload PDFs or text files → extract → chunk → embed  
- 🧠 **Contextual AI Answers:** Combines vector search + LLM for document-aware answers  
- 🗄️ **Chroma Vector DB:** Efficient, scalable embedding storage per user  
- 🐳 **Dockerized Architecture:** Fully containerized with FastAPI + Vector DB  
- 💬 **Transparent Responses:** Each answer includes document source references  

---

## ⚙️ System Workflow  

1️⃣ User logs in → gets a unique ID
2️⃣ Upload document → extract text → chunk → embed → store in user DB
3️⃣ Ask query → generate embedding → search vector DB → retrieve top chunks
4️⃣ Feed retrieved context to Meta LLaMA via Cerebras → generate AI answer with sources

---

## 🧩 Architecture  

[ Mobile / Web Client ]
|
v
[ FastAPI Backend ]
|
v
[ User-wise Chroma DB ]
|
v
[ Cerebras LLaMA Model ]


**Components:**  
- **FastAPI** → REST APIs for authentication, file upload, and chat queries  
- **Chroma DB** → Stores user-specific embeddings  
- **PyPDF2** → Extracts text from uploaded PDFs  
- **Sentence Transformers (MiniLM-L6-v2)** → Generates embeddings for semantic search  
- **Cerebras SDK + Meta LLaMA 17B** → Produces contextual answers  
