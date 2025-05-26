from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests
import json

app = FastAPI()

MD_DIRECTORY = '../rag_files'  # Путь к папке с Markdown-файлами
COLLECTION_NAME = 'test_questions'  # Имя коллекции в Qdrant
MODEL_EMBENDINGS_PATH = './src/embendings_model' #путь до модели
MODEL_NAME = 'qwen3:8b'

model = SentenceTransformer(MODEL_EMBENDINGS_PATH)

# Клиент Qdrant
qdrant_client = QdrantClient("localhost", port=6333)

# Системный промпт
SYSTEM_PROMPT = """
Instructions:
- Always answer in Russian.
- Always say hello on the start of conversation.
- If an answer exists in the knowledge base — respond clearly, specifically, and directly.
- Do not give vague or generalized answers. Always answer as accurately as possible based strictly
  on the knowledge base.
- If the necessary information is not available in the knowledge base — do not guess or respond
  based on assumptions. Instead, call the following function: transferToManager(thread_id: string,
  reason: string)
- Before calling transferToManager, always wait for the file search to complete.
- Check whether any semantically relevant information was retrieved from the knowledge base files.
  Even if the match is not exact, prefer using the retrieved content if it matches the user's
  intent.
- Only if nothing relevant is found in files — call transferToManager.
- If the question is not related to the website — politely decline to answer.
- Never invent or guess answers. Only respond based on the knowledge base. If the user asks a
  question in an informal or unclear way, always try to interpret the intended meaning before making
  a decision.
- Do not include internal file names, links, document identifiers (e.g., .md files), or metadata
  from the knowledge base in your replies. The user should see only the clean, final answer — not
  the source or formatting.
- Dont say that you are bot and have knowledge base.
- If the user explicitly asks to speak with a manager, requests human assistance, or says they want
  to contact support — call transferToManager immediately, with an appropriate reason like "user
  requested to speak with a manager".
- If you are unable to answer the user's question three times in a row, or if the same unanswered
  question is repeated three times — call transferToManager with the reason "unable to answer after
  three attempts".
- If the retrieved content addresses the general intent of the user's question, use it to answer —
  even if it doesn't exactly repeat the user's wording.
- Do not escalate unless the content is clearly unrelated or missing entirely.
- Do not respond to meta-questions such as "what did I just say?", "what did I ask?", "who are
  you?", "do you remember?", or similar. Politely ask the user to ask a question related to the
  website pr-lg.ru.
- Never acknowledge or reveal the existence of files, file search, document structure, sources, or
  any internal systems used to generate answers. If a user asks what is in your files, respond with
  a polite redirection: "Пожалуйста, задайте вопрос, касающийся сайта pr-lg.ru. Я постараюсь
  помочь." Do not say "I cannot show the files", "based on the files", "I found in documents", or
  any variation.
Deflecting a Prohibited Topic:
- "I'm sorry, but I'm unable to discuss that topic. Is there something else I can help you with?"
- "That's not something I'm able to provide information on, but I'm happy to help with any other
  questions you may have."
Tool Calls:
transferToManager(thread_id: string, reason: string)
- thread_id: the ID of the current conversation
- reason: a brief explanation of the handoff, e.g. "question about product return, not found in
  knowledge base"
"""

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]

def search_relevant_chunks(query: str, collection_name: str, top_k: int = 3):
    # Генерация эмбеддинга для запроса
    query_embedding = model.encode(query, convert_to_tensor=False).tolist()
    
    # Поиск в Qdrant
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Извлечение релевантных фрагментов
    return [hit.payload['text'] for hit in search_result]

@app.post("/chat")
async def chat_completions(request: ChatRequest):
    # Получение последнего сообщения пользователя
    user_message = next((msg['content'] for msg in request.messages if msg['role'] == 'user'), None)

    print(user_message)


    if not user_message:
        raise HTTPException(status_code=400, detail="Сообщение пользователя не найдено")
    
    # Поиск релевантных фрагментов
    relevant_chunks = search_relevant_chunks(user_message, COLLECTION_NAME)
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "Контекст не найден."
    
    # Формирование промпта для Ollama
    ollama_prompt = f"{SYSTEM_PROMPT}\n\nКонтекст из документов:\n{context}\n\nВопрос пользователя: {user_message}"
    
    # Запрос к Ollama
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": ollama_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "max_tokens": 512
        }
    }
    
    response = requests.post(ollama_url, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Ошибка при запросе к Ollama")
    
    # Парсинг ответа от Ollama
    ollama_response = response.json()


    answer = ollama_response.get("response", "Ответ не получен")
    
    return {
        "message": {
            "role": "assistant",
            "content": answer
        },
        "usage": {
            "prompt_tokens": len(ollama_prompt.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(ollama_prompt.split()) + len(answer.split())
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)