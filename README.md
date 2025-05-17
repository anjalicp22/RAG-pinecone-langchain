# RAG-pinecone-langchain-app
DeepSeek R1 Chatbot is a RAG-based web assistant that answers questions grounded in the DeepSeek R1 paper using Cohere for embeddings/LLM, Pinecone for vector search, and a Flask + HTML/JS/CSS frontend.

Reference: https://www.youtube.com/watch?v=LhnCsygAvzY 

# 📚 DeepSeek R1 Chatbot – Research Assistant (RAG-based)
This project is an RAG (Retrieval-Augmented Generation)-powered research assistant built on the DeepSeek R1 paper. It enables users to interact with the paper using natural language questions, leveraging Cohere for embedding and generation and Pinecone for vector search.

# 🔧 Tech Stack
•	Frontend: HTML5, CSS3 (custom + Bootstrap 5)

•	Backend: Python, Flask

•	LLM: Cohere via langchain-cohere

•	Vector Store: Pinecone

•	Embedding Model: Cohere embed-v4.0

•	Dataset: [HuggingFace DeepSeek R1 (chunked)](https://huggingface.co/datasets/jamescalam/deepseek-r1-paper-chunked)

# 🚀 Features
•	RAG pipeline to answer user queries grounded in research context

•	Clean, responsive web UI with typing indicator and animations

•	Abort controller for timeout handling on the frontend


