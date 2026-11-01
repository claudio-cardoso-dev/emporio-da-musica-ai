"""FastAPI application with OpenAPI/Swagger documentation and an embedded glassmorphism Web Chat UI."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.services.chat_service import ChatService


class ChatRequest(BaseModel):
    """Request payload for sending a message to the Empório da Música AI Agent."""

    message: str = Field(..., description="Customer message in any language (PT, EN, ES).")
    session_id: str = Field(
        default="web_session_01",
        description="Unique session identifier for multi-turn conversation memory.",
    )


class ChatResponse(BaseModel):
    """Structured response returned by the conversational AI Agent."""

    session_id: str
    reply: str
    language: str
    confidence: float
    model_used: str
    tools_called: List[Dict[str, Any]]
    logged_failed_intent: bool


class ResetSessionRequest(BaseModel):
    """Payload to clear conversation history for a session."""

    session_id: str = Field(default="web_session_01")


WEB_CHAT_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Empório da Música — Assistente Virtual IA</title>
  <style>
    :root {
      --bg-dark: #0f172a;
      --card-bg: rgba(30, 41, 59, 0.82);
      --accent: #f59e0b;
      --accent-hover: #d97706;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border: rgba(255, 255, 255, 0.1);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
    body {
      background: radial-gradient(circle at top right, #1e293b, #0f172a 70%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 16px;
    }
    .chat-container {
      width: 100%;
      max-width: 780px;
      height: 88vh;
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 20px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.6);
    }
    .header {
      padding: 18px 24px;
      background: rgba(15, 23, 42, 0.75);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .avatar {
      width: 44px; height: 44px; border-radius: 12px;
      background: linear-gradient(135deg, #f59e0b, #ea580c);
      display: flex; align-items: center; justify-content: center;
      font-size: 22px; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.35);
    }
    .brand-info h1 { font-size: 1.1rem; font-weight: 700; color: #fff; }
    .brand-info p { font-size: 0.8rem; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
    .pulse {
      width: 8px; height: 8px; background: #10b981; border-radius: 50%;
      box-shadow: 0 0 8px #10b981;
    }
    .header-actions { display: flex; gap: 8px; }
    .btn-ghost {
      background: rgba(255,255,255,0.06); color: var(--text-muted);
      border: 1px solid var(--border); padding: 7px 12px; border-radius: 8px;
      cursor: pointer; font-size: 0.8rem; transition: all 0.2s; text-decoration: none;
    }
    .btn-ghost:hover { color: #fff; background: rgba(255,255,255,0.12); }
    .messages {
      flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px;
    }
    .msg { max-width: 82%; padding: 13px 16px; border-radius: 14px; line-height: 1.5; font-size: 0.94rem; white-space: pre-wrap; }
    .msg.bot {
      align-self: flex-start; background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--border); border-bottom-left-radius: 4px;
    }
    .msg.user {
      align-self: flex-end; background: linear-gradient(135deg, #d97706, #b45309);
      color: #fff; border-bottom-right-radius: 4px;
    }
    .meta-badge {
      margin-top: 8px; font-size: 0.72rem; color: var(--text-muted);
      border-top: 1px solid rgba(255,255,255,0.07); padding-top: 5px;
    }
    .chips {
      padding: 10px 20px; display: flex; gap: 8px; overflow-x: auto;
      border-top: 1px solid var(--border); background: rgba(15, 23, 42, 0.45);
    }
    .chip {
      white-space: nowrap; background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.3); color: #fcd34d;
      padding: 6px 12px; border-radius: 999px; font-size: 0.78rem; cursor: pointer;
      transition: all 0.2s;
    }
    .chip:hover { background: rgba(245, 158, 11, 0.25); }
    .input-area {
      padding: 16px 20px; background: rgba(15, 23, 42, 0.85);
      border-top: 1px solid var(--border); display: flex; gap: 10px;
    }
    .input-area input {
      flex: 1; background: rgba(30, 41, 59, 0.9); border: 1px solid var(--border);
      color: #fff; padding: 12px 16px; border-radius: 12px; outline: none; font-size: 0.95rem;
    }
    .input-area input:focus { border-color: var(--accent); }
    .btn-send {
      background: var(--accent); color: #0f172a; font-weight: 700;
      border: none; padding: 0 22px; border-radius: 12px; cursor: pointer; transition: 0.2s;
    }
    .btn-send:hover { background: var(--accent-hover); color: #fff; }
  </style>
</head>
<body>
  <div class="chat-container">
    <div class="header">
      <div class="brand">
        <div class="avatar">🎸</div>
        <div class="brand-info">
          <h1>Empório da Música</h1>
          <p><span class="pulse"></span> Assistente IA Online • Campo Grande/MS</p>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn-ghost" onclick="resetChat()">🔄 Reiniciar</button>
        <a class="btn-ghost" href="/docs" target="_blank">📄 Swagger API</a>
      </div>
    </div>
    <div class="messages" id="messages">
      <div class="msg bot">Olá! Bem-vindo(a) à <strong>Empório da Música</strong>! 🎶
Sou o assistente virtual da loja. Posso ajudar você a consultar nosso catálogo de instrumentos musicais, preços e promoções, simular parcelamentos e frete, acompanhar pedidos ou tirar dúvidas sobre nossas políticas de troca e garantia. Como posso te ajudar hoje?</div>
    </div>
    <div class="chips">
      <span class="chip" onclick="sendQuick('Quais opções de violões disponíveis custando até R$1000?')">🎸 Violões até R$1000</span>
      <span class="chip" onclick="sendQuick('Quanto custa o Takamine GD20?')">💰 Preço Takamine GD20</span>
      <span class="chip" onclick="sendQuick('Qual o endereço e horário de funcionamento da loja?')">📍 Endereço e Horário</span>
      <span class="chip" onclick="sendQuick('Me arrependi da minha compra online, posso devolver meu pedido?')">🔄 Política de Devolução</span>
      <span class="chip" onclick="sendQuick('Qual o status do pedido #8?')">📦 Rastrear Pedido #8</span>
    </div>
    <div class="input-area">
      <input type="text" id="userInput" placeholder="Digite sua mensagem em Português, Inglês ou Espanhol..." onkeydown="if(event.key==='Enter') sendMessage()" />
      <button class="btn-send" onclick="sendMessage()">Enviar</button>
    </div>
  </div>
  <script>
    const sessionId = "web_" + Math.random().toString(36).substring(2, 9);
    const messagesEl = document.getElementById("messages");
    const inputEl = document.getElementById("userInput");

    function appendMsg(text, role, meta = null) {
      const div = document.createElement("div");
      div.className = "msg " + role;
      div.textContent = text;
      if (meta) {
        const badge = document.createElement("div");
        badge.className = "meta-badge";
        const tools = (meta.tools_called || []).map(t => t.tool).join(", ") || "direct";
        badge.textContent = `Model: ${meta.model_used} | Lang: ${meta.language} | Tools: ${tools}`;
        div.appendChild(badge);
      }
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function sendQuick(text) {
      inputEl.value = text;
      sendMessage();
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = "";
      appendMsg(text, "user");
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: sessionId })
        });
        const data = await res.json();
        appendMsg(data.reply, "bot", data);
      } catch (e) {
        appendMsg("Erro ao comunicar com o servidor.", "bot");
      }
    }

    async function resetChat() {
      await fetch("/api/session/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      messagesEl.innerHTML = '<div class="msg bot">Sessão reiniciada! Como posso ajudar você hoje na Empório da Música? 🎶</div>';
    }
  </script>
</body>
</html>
"""


def create_app(db_path: Optional[Path] = None, pdf_path: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Empório da Música — AI Customer Support Agent API",
        description=(
            "Conversational AI Engineer Challenge (Artefact). "
            "Hybrid Architecture combining LLM Function Calling (Qwen3-32B primary + Llama fallbacks), "
            "SQLite Operational Repository, Deterministic Pricing Engine, and Semantic PDF RAG."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    chat_service = ChatService(db_path=db_path, pdf_path=pdf_path)

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    def serve_web_ui() -> HTMLResponse:
        """Render the embedded Glassmorphism Web Chat interface."""
        return HTMLResponse(content=WEB_CHAT_HTML)

    @app.get("/api/health", tags=["System"])
    def health_check() -> Dict[str, Any]:
        """Return service health, configured LLM fallback chain, and database status."""
        return {
            "status": "ok",
            "store": "Empório da Música (Campo Grande/MS)",
            "primary_model": settings.primary_model,
            "fallback_models": [settings.fallback_model_1, settings.fallback_model_2],
        }

    @app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
    def chat_endpoint(payload: ChatRequest) -> ChatResponse:
        """Send a message to the Empório da Música virtual assistant."""
        if not payload.message or not payload.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        result = chat_service.process_message(
            user_message=payload.message,
            session_id=payload.session_id,
        )
        return ChatResponse(
            session_id=payload.session_id,
            reply=result["reply"],
            language=result["language"],
            confidence=result["confidence"],
            model_used=result["model_used"],
            tools_called=result.get("tools_called", []),
            logged_failed_intent=result.get("logged_failed_intent", False),
        )

    @app.get("/api/session/{session_id}/history", tags=["Chat"])
    def get_session_history(session_id: str) -> Dict[str, Any]:
        """Retrieve conversation history for a given session_id."""
        return {
            "session_id": session_id,
            "history": chat_service.get_history(session_id=session_id),
        }

    @app.post("/api/session/reset", tags=["Chat"])
    def reset_session(payload: ResetSessionRequest) -> Dict[str, Any]:
        """Clear conversation history for the specified session_id."""
        chat_service.clear_session(payload.session_id)
        return {"status": "cleared", "session_id": payload.session_id}

    @app.get("/api/failed-intents", tags=["Observability"])
    def list_failed_intents() -> Dict[str, Any]:
        """List logged low-confidence or ambiguous user interactions for NLU auditing."""
        failures = chat_service.agent.failed_intent_repo.get_all_failures(limit=50)
        return {"count": len(failures), "items": failures}

    return app


app = create_app()
