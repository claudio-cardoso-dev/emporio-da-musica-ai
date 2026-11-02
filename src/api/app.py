"""FastAPI application with OpenAPI/Swagger documentation and an embedded glassmorphism Web Chat UI with Interactive Product Carousel & Cart."""

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
    products: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured product objects returned by catalog tools for Carousel rendering.",
    )
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
      --card-bg: rgba(30, 41, 59, 0.85);
      --accent: #f59e0b;
      --accent-hover: #d97706;
      --emerald: #10b981;
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
      padding: 14px;
    }
    .chat-container {
      width: 100%;
      max-width: 920px;
      height: 92vh;
      background: var(--card-bg);
      backdrop-filter: blur(18px);
      border: 1px solid var(--border);
      border-radius: 20px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.65);
      position: relative;
    }
    .header {
      padding: 16px 24px;
      background: rgba(15, 23, 42, 0.82);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
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
      width: 8px; height: 8px; background: var(--emerald); border-radius: 50%;
      box-shadow: 0 0 8px var(--emerald);
    }
    .header-actions { display: flex; gap: 8px; align-items: center; }
    .btn-ghost {
      background: rgba(255,255,255,0.06); color: var(--text-muted);
      border: 1px solid var(--border); padding: 8px 13px; border-radius: 10px;
      cursor: pointer; font-size: 0.82rem; transition: all 0.2s; text-decoration: none;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .btn-ghost:hover { color: #fff; background: rgba(255,255,255,0.13); }
    .btn-cart {
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid rgba(245, 158, 11, 0.45);
      color: #fcd34d;
      font-weight: 600;
      position: relative;
    }
    .cart-badge {
      background: var(--accent);
      color: #0f172a;
      font-size: 0.72rem;
      font-weight: 800;
      padding: 2px 6px;
      border-radius: 999px;
    }
    .messages {
      flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px;
    }
    .msg-wrapper {
      display: flex; flex-direction: column; gap: 10px; width: 100%;
    }
    .msg { max-width: 85%; padding: 14px 18px; border-radius: 14px; line-height: 1.55; font-size: 0.93rem; }
    .msg.bot {
      align-self: flex-start; background: rgba(15, 23, 42, 0.88);
      border: 1px solid var(--border); border-bottom-left-radius: 4px;
    }
    .msg.user {
      align-self: flex-end; background: linear-gradient(135deg, #d97706, #b45309);
      color: #fff; border-bottom-right-radius: 4px; white-space: pre-wrap;
    }
    /* Formatted Markdown Elements inside Bot Messages */
    .md-content h2, .md-content h3, .md-content h4 {
      color: #fcd34d; margin: 10px 0 6px 0; font-size: 1rem;
    }
    .md-content p { margin-bottom: 8px; }
    .md-content ul, .md-content ol { padding-left: 20px; margin-bottom: 8px; }
    .md-content li { margin-bottom: 4px; }
    .md-table-wrap { overflow-x: auto; margin: 10px 0; border-radius: 8px; border: 1px solid var(--border); }
    .md-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
    .md-table th { background: rgba(245, 158, 11, 0.15); color: #fcd34d; text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
    .md-table td { padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .md-table tr:last-child td { border-bottom: none; }

    .meta-badge {
      margin-top: 10px; font-size: 0.72rem; color: var(--text-muted);
      border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;
    }

    /* Product Carousel Styles */
    .carousel-section {
      width: 100%;
      margin-top: 4px;
    }
    .carousel-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 8px; padding: 0 4px;
    }
    .carousel-title {
      font-size: 0.82rem; font-weight: 700; color: #fcd34d;
      text-transform: uppercase; letter-spacing: 0.04em;
    }
    .carousel-nav { display: flex; gap: 6px; }
    .nav-btn {
      width: 28px; height: 28px; border-radius: 8px;
      background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border);
      color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center;
      font-size: 0.85rem; transition: 0.2s;
    }
    .nav-btn:hover { border-color: var(--accent); color: var(--accent); }
    .carousel-track {
      display: flex;
      gap: 14px;
      overflow-x: auto;
      scroll-behavior: smooth;
      padding: 4px 2px 12px 2px;
      scrollbar-width: thin;
      scrollbar-color: rgba(245, 158, 11, 0.4) transparent;
    }
    .carousel-track::-webkit-scrollbar { height: 6px; }
    .carousel-track::-webkit-scrollbar-thumb { background: rgba(245, 158, 11, 0.4); border-radius: 999px; }

    .prod-card {
      min-width: 270px;
      max-width: 270px;
      height: 375px;
      perspective: 1000px;
      position: relative;
      background: transparent;
      border: none;
      padding: 0;
      flex-shrink: 0;
      transition: transform 0.2s ease;
    }
    .prod-card:hover {
      transform: translateY(-4px);
    }
    .card-inner {
      position: relative;
      width: 100%;
      height: 100%;
      transform-style: preserve-3d;
      transition: transform 0.65s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 16px;
    }
    .card-inner.is-flipped {
      transform: rotateY(180deg);
    }
    .card-face {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      -webkit-backface-visibility: hidden;
      backface-visibility: hidden;
      border-radius: 16px;
      padding: 13px 14px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      border: 1px solid rgba(255, 255, 255, 0.12);
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    }
    .card-front {
      background: linear-gradient(160deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.92));
      z-index: 2;
      transform: rotateY(0deg);
    }
    .card-front:hover {
      border-color: rgba(245, 158, 11, 0.55);
    }
    .card-back {
      background: linear-gradient(160deg, rgba(20, 30, 48, 0.98), rgba(15, 23, 42, 0.96));
      border: 1px solid rgba(245, 158, 11, 0.45);
      transform: rotateY(180deg);
      z-index: 1;
    }
    .prod-top-badges {
      display: flex; justify-content: space-between; align-items: center; gap: 6px;
    }
    .cat-tag {
      font-size: 0.72rem; font-weight: 600; color: #94a3b8; background: rgba(255,255,255,0.07);
      padding: 3px 8px; border-radius: 6px; white-space: nowrap;
      border: 1px solid transparent;
    }
    .cat-tag-clickable {
      font-size: 0.72rem; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.14);
      border: 1px solid rgba(56, 189, 248, 0.45); padding: 3px 8px; border-radius: 6px;
      cursor: pointer; display: inline-flex; align-items: center; gap: 4px; transition: all 0.2s ease;
      font-family: inherit; white-space: nowrap;
    }
    .cat-tag-clickable:hover {
      background: rgba(56, 189, 248, 0.3); border-color: #38bdf8; color: #fff; transform: scale(1.04);
    }
    .btn-desc-flip {
      font-size: 0.7rem; font-weight: 700; color: #fcd34d; background: rgba(245, 158, 11, 0.16);
      border: 1px solid rgba(245, 158, 11, 0.4); padding: 3px 8px; border-radius: 6px;
      cursor: pointer; display: inline-flex; align-items: center; gap: 3px; transition: all 0.2s ease;
      white-space: nowrap; font-family: inherit;
    }
    .btn-desc-flip:hover {
      background: rgba(245, 158, 11, 0.32); border-color: var(--accent); color: #fff; transform: scale(1.04);
    }
    .card-back-body {
      flex: 1; display: flex; flex-direction: column; gap: 6px; margin-top: 6px; overflow: hidden;
    }
    .card-back-title {
      font-size: 0.86rem; font-weight: 700; color: #fcd34d; line-height: 1.3;
    }
    .card-back-meta {
      display: flex; flex-wrap: wrap; gap: 4px;
    }
    .meta-pill {
      font-size: 0.68rem; background: rgba(255, 255, 255, 0.07); color: #cbd5e1;
      padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .card-back-desc-box {
      flex: 1; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px; padding: 8px 10px; overflow-y: auto;
      scrollbar-width: thin; scrollbar-color: rgba(245, 158, 11, 0.4) transparent;
      margin-top: 2px; max-height: 140px;
    }
    .card-back-desc-box::-webkit-scrollbar { width: 4px; }
    .card-back-desc-box::-webkit-scrollbar-thumb { background: rgba(245, 158, 11, 0.4); border-radius: 999px; }
    .card-back-desc-heading {
      font-size: 0.68rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;
      letter-spacing: 0.04em; margin-bottom: 4px;
    }
    .card-back-desc-content {
      font-size: 0.76rem; color: #e2e8f0; line-height: 1.4; white-space: pre-wrap;
    }
    .card-back-footer {
      display: flex; flex-direction: column; gap: 6px; margin-top: 6px; padding-top: 6px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .card-back-price-box {
      display: flex; justify-content: space-between; align-items: baseline;
    }
    .card-back-price {
      font-size: 0.95rem; font-weight: 800; color: #fcd34d;
    }
    .card-back-pix {
      font-size: 0.72rem; font-weight: 600; color: #6ee7b7;
    }
    .btn-return-card {
      width: 100%; padding: 7px 10px; border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.15); background: rgba(255, 255, 255, 0.08);
      color: #fff; font-weight: 700; font-size: 0.78rem; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 6px;
      transition: 0.2s; font-family: inherit;
    }
    .btn-return-card:hover {
      background: rgba(255, 255, 255, 0.18); border-color: #38bdf8; color: #38bdf8;
    }
    .promo-tag {
      font-size: 0.68rem; font-weight: 700; background: rgba(239, 68, 68, 0.2);
      color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4);
      padding: 2px 7px; border-radius: 999px;
    }
    .stock-tag {
      font-size: 0.68rem; font-weight: 600; padding: 2px 7px; border-radius: 999px;
    }
    .stock-ok { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; }
    .stock-out { background: rgba(239, 68, 68, 0.18); color: #fca5a5; }
    .prod-icon-box {
      height: 58px; border-radius: 12px;
      background: radial-gradient(circle, rgba(245, 158, 11, 0.16), rgba(15, 23, 42, 0.6));
      display: flex; align-items: center; justify-content: center;
      font-size: 28px; border: 1px solid rgba(255,255,255,0.05);
    }
    .prod-title {
      font-size: 0.9rem; font-weight: 700; color: #fff; line-height: 1.3;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
      min-height: 2.35em;
    }
    .prod-desc {
      font-size: 0.75rem; color: var(--text-muted); line-height: 1.35;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .price-box {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 10px;
      padding: 8px 10px;
    }
    .old-price {
      font-size: 0.74rem; color: #94a3b8; text-decoration: line-through;
    }
    .main-price {
      font-size: 1.08rem; font-weight: 800; color: #fcd34d;
    }
    .pix-price {
      font-size: 0.75rem; color: #6ee7b7; font-weight: 600; margin-top: 2px;
    }
    .card-actions {
      display: flex; flex-direction: column; gap: 7px; margin-top: 2px;
    }
    .qty-row {
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border);
      border-radius: 9px; padding: 3px 6px;
    }
    .qty-btn {
      width: 26px; height: 26px; border-radius: 6px; border: none;
      background: rgba(255,255,255,0.08); color: #fff; font-weight: 700;
      cursor: pointer; transition: 0.15s;
    }
    .qty-btn:hover:not(:disabled) { background: var(--accent); color: #0f172a; }
    .qty-val { font-size: 0.85rem; font-weight: 700; color: #fff; min-width: 24px; text-align: center; }
    .btn-add-cart {
      width: 100%; padding: 9px 10px; border-radius: 9px; border: none;
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #0f172a; font-weight: 800; font-size: 0.8rem; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 6px;
      transition: 0.2s;
    }
    .btn-add-cart:hover:not(:disabled) { filter: brightness(1.1); transform: scale(1.01); }
    .btn-add-cart:disabled {
      background: rgba(148, 163, 184, 0.2); color: #64748b; cursor: not-allowed;
    }

    /* Cart Drawer */
    .cart-drawer {
      position: absolute; top: 70px; right: 20px; width: 350px; max-height: 75vh;
      background: rgba(15, 23, 42, 0.97); backdrop-filter: blur(20px);
      border: 1px solid rgba(245, 158, 11, 0.45); border-radius: 16px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.75);
      display: none; flex-direction: column; z-index: 50; overflow: hidden;
    }
    .cart-drawer.open { display: flex; }
    .cart-drawer-header {
      padding: 14px 16px; border-bottom: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: center;
      font-weight: 700; color: #fcd34d;
    }
    .cart-items {
      padding: 12px 16px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 10px;
    }
    .cart-item {
      display: flex; justify-content: space-between; align-items: center;
      background: rgba(30, 41, 59, 0.75); padding: 9px 12px; border-radius: 10px;
      border: 1px solid var(--border); font-size: 0.82rem;
    }
    .cart-item-info { flex: 1; padding-right: 8px; }
    .cart-item-title { font-weight: 700; color: #fff; }
    .cart-item-sub { color: var(--text-muted); font-size: 0.75rem; margin-top: 2px; }
    .btn-rm {
      background: rgba(239, 68, 68, 0.18); color: #fca5a5; border: none;
      width: 24px; height: 24px; border-radius: 6px; cursor: pointer; font-weight: 700;
    }
    .cart-footer {
      padding: 14px 16px; border-top: 1px solid var(--border);
      background: rgba(30, 41, 59, 0.6); display: flex; flex-direction: column; gap: 8px;
    }
    .cart-total-row { display: flex; justify-content: space-between; font-size: 0.88rem; }
    .cart-total-row.pix { color: #6ee7b7; font-weight: 700; }
    .btn-checkout {
      background: var(--emerald); color: #0f172a; font-weight: 800;
      border: none; padding: 10px; border-radius: 10px; cursor: pointer; font-size: 0.85rem;
      transition: 0.2s;
    }
    .btn-checkout:hover { filter: brightness(1.1); }

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
      padding: 15px 20px; background: rgba(15, 23, 42, 0.85);
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
        <button class="btn-ghost btn-cart" onclick="toggleCart()">
          🛒 Carrinho <span class="cart-badge" id="cartCount">0</span>
        </button>
        <button class="btn-ghost" onclick="resetChat()">🔄 Reiniciar</button>
        <a class="btn-ghost" href="/docs" target="_blank">📄 Swagger API</a>
      </div>
    </div>

    <!-- Shopping Cart Floating Drawer -->
    <div class="cart-drawer" id="cartDrawer">
      <div class="cart-drawer-header">
        <span>🛒 Seu Carrinho Musical</span>
        <button class="btn-ghost" style="padding:3px 8px;" onclick="toggleCart()">✕</button>
      </div>
      <div class="cart-items" id="cartItemsList">
        <p style="color:#94a3b8; font-size:0.82rem; text-align:center; padding:16px 0;">Seu carrinho está vazio.</p>
      </div>
      <div class="cart-footer" id="cartFooter" style="display:none;">
        <div class="cart-total-row">
          <span>Total Cartão (até 12x):</span>
          <strong id="cartTotalRegular">R$ 0,00</strong>
        </div>
        <div class="cart-total-row pix">
          <span>Total à vista no PIX:</span>
          <strong id="cartTotalPix">R$ 0,00</strong>
        </div>
        <button class="btn-checkout" onclick="simulateCartWithAgent()">
          ⚡ Simular Pagamento e Frete no Chat
        </button>
      </div>
    </div>

    <div class="messages" id="messages">
      <div class="msg-wrapper">
        <div class="msg bot">
          <div class="md-content">Olá! Bem-vindo(a) à <strong>Empório da Música</strong>! 🎶<br/>Sou o assistente virtual da loja. Quando você buscar instrumentos, exibirei um <strong>carrossel interativo de produtos</strong> onde você pode escolher a quantidade e adicionar ao carrinho para simular parcelamentos e frete. Como posso te ajudar hoje?</div>
        </div>
      </div>
    </div>

    <div class="chips">
      <span class="chip" onclick="sendQuick('Quais opções de violões disponíveis custando até R$1000?')">🎸 Violões até R$1000</span>
      <span class="chip" onclick="sendQuick('Quais opções de ukuleles disponíveis?')">🏝️ Ver Ukuleles</span>
      <span class="chip" onclick="sendQuick('Quais produtos estão em promoção ativa hoje?')">🔥 Produtos em Promoção</span>
      <span class="chip" onclick="sendQuick('Quanto custa o Takamine GD20?')">💰 Preço Takamine GD20</span>
      <span class="chip" onclick="sendQuick('Qual o endereço e horário de funcionamento da loja?')">📍 Endereço e Horário</span>
      <span class="chip" onclick="sendQuick('Quais informações posso usar para consultar meu pedido ou rastreio?')">📦 Consultar Pedido / Rastreio</span>
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
    let cart = [];

    function formatBRL(val) {
      return Number(val || 0).toLocaleString("pt-BR", { style: "currency", currency: "BRL" });
    }

    function getCategoryEmoji(catName) {
      const c = (catName || "").toLowerCase();
      if (c.includes("guitarra")) return "🎸";
      if (c.includes("viol")) return "🪕";
      if (c.includes("baixo")) return "🎸";
      if (c.includes("bateria") || c.includes("percuss")) return "🥁";
      if (c.includes("teclado") || c.includes("piano")) return "🎹";
      if (c.includes("ukulele")) return "🏝️";
      if (c.includes("sopro")) return "🎷";
      if (c.includes("orquestra")) return "🎻";
      return "🎵";
    }

    function escapeHtml(str) {
      return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function formatInlineMd(text) {
      let s = escapeHtml(text);
      s = s.replace(/~~(.+?)~~/g, "<del>$1</del>");
      s = s.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>");
      s = s.replace(/\\*(.+?)\\*/g, "<em>$1</em>");
      s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
      return s;
    }

    function renderMarkdown(md) {
      const lines = String(md || "").split("\\n");
      let html = "";
      let i = 0;

      while (i < lines.length) {
        const line = lines[i].trim();

        // Markdown Table Detection
        if (line.startsWith("|") && i + 1 < lines.length && lines[i + 1].trim().match(/^\\|[\\s\\-:|]+\\|$/)) {
          const headers = line.split("|").slice(1, -1).map(h => formatInlineMd(h.trim()));
          i += 2;
          let rowsHtml = "";
          while (i < lines.length && lines[i].trim().startsWith("|")) {
            const cols = lines[i].trim().split("|").slice(1, -1).map(c => formatInlineMd(c.trim()));
            rowsHtml += "<tr>" + cols.map(c => `<td>${c}</td>`).join("") + "</tr>";
            i++;
          }
          html += `<div class="md-table-wrap"><table class="md-table"><thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rowsHtml}</tbody></table></div>`;
          continue;
        }

        if (line.startsWith("### ")) {
          html += `<h3>${formatInlineMd(line.slice(4))}</h3>`;
        } else if (line.startsWith("## ")) {
          html += `<h2>${formatInlineMd(line.slice(3))}</h2>`;
        } else if (line.startsWith("- ") || line.startsWith("• ")) {
          html += `<ul><li>${formatInlineMd(line.slice(2))}</li></ul>`;
        } else if (line === "---") {
          html += `<hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin:8px 0;" />`;
        } else if (line.length > 0) {
          html += `<p>${formatInlineMd(line)}</p>`;
        }
        i++;
      }
      return html;
    }

    function buildProductCarousel(products) {
      if (!products || products.length === 0) return null;

      const section = document.createElement("div");
      section.className = "carousel-section";
      const trackId = "track_" + Math.random().toString(36).substring(2, 8);

      section.innerHTML = `
        <div class="carousel-header">
          <span class="carousel-title">🎸 Instrumentos Encontrados (${products.length}) — Arraste ou use as setas</span>
          <div class="carousel-nav">
            <button class="nav-btn" onclick="document.getElementById('${trackId}').scrollBy({left: -520, behavior: 'smooth'})">◀</button>
            <button class="nav-btn" onclick="document.getElementById('${trackId}').scrollBy({left: 520, behavior: 'smooth'})">▶</button>
          </div>
        </div>
      `;

      const track = document.createElement("div");
      track.className = "carousel-track";
      track.id = trackId;

      products.forEach(p => {
        const card = document.createElement("div");
        card.className = "prod-card";
        const pid = p.product_id;
        const maxStock = Number(p.stock_quantity || 0);
        const inStock = Boolean(p.in_stock && maxStock > 0);
        const activePromo = p.active_promotion;
        const emoji = getCategoryEmoji(p.category_name);

        let statusBadge = "";
        if (p.status === "discontinued") {
          statusBadge = `<span class="stock-tag stock-out">Descontinuado</span>`;
        } else if (p.status === "coming_soon") {
          statusBadge = `<span class="stock-tag stock-out">Em Breve</span>`;
        } else if (!inStock) {
          statusBadge = `<span class="stock-tag stock-out">Sem Estoque</span>`;
        } else {
          statusBadge = `<span class="stock-tag stock-ok">${maxStock} em estoque</span>`;
        }

        const promoBadge = activePromo
          ? `<span class="promo-tag">🔥 -${activePromo.discount_percent}% OFF</span>`
          : "";

        const priceHtml = activePromo
          ? `<div class="old-price">De: ${formatBRL(p.price_brl)}</div>
             <div class="main-price">${formatBRL(p.effective_price_brl)}</div>
             <div class="pix-price">Promoção ativa (PIX não cumulativo)</div>`
          : `<div class="main-price">${formatBRL(p.effective_price_brl)}</div>
             <div class="pix-price">⚡ ${formatBRL(p.pix_price_brl)} no PIX (5% off)</div>`;

        card.innerHTML = `
          <div class="card-inner" id="card_inner_${pid}">
            <!-- FACE FRONTAL -->
            <div class="card-face card-front">
              <div>
                <div class="prod-top-badges">
                  <span class="cat-tag">${escapeHtml(p.category_name || "Instrumento")}</span>
                  <button class="btn-desc-flip" onclick="flipCardToBack(${pid})" title="Ver descrição completa">
                    📖 Descrição
                  </button>
                  <div style="display:flex; gap:4px; align-items:center;">
                    ${promoBadge}
                    ${statusBadge}
                  </div>
                </div>
                <div class="prod-icon-box">${emoji}</div>
                <div class="prod-title" title="${escapeHtml(p.name)}">${escapeHtml(p.name)}</div>
              </div>
              <div>
                <div class="price-box">
                  ${priceHtml}
                  <div style="margin-top:4px; font-size:0.7rem; color:#94a3b8;">${promoBadge ? statusBadge : ""}</div>
                </div>
                <div class="card-actions">
                  <div class="qty-row">
                    <span style="font-size:0.74rem; color:#94a3b8; padding-left:4px;">Qtd:</span>
                    <div style="display:flex; align-items:center; gap:6px;">
                      <button class="qty-btn" onclick="changeCardQty(${pid}, -1, ${maxStock})" ${!inStock ? "disabled" : ""}>−</button>
                      <span class="qty-val" id="qty_${pid}">1</span>
                      <button class="qty-btn" onclick="changeCardQty(${pid}, 1, ${maxStock})" ${!inStock ? "disabled" : ""}>+</button>
                    </div>
                  </div>
                  <button class="btn-add-cart" id="btn_add_${pid}" ${!inStock ? "disabled" : ""}>
                    ${inStock ? "🛒 Adicionar ao Carrinho" : "Indisponível"}
                  </button>
                </div>
              </div>
            </div>

            <!-- FACE TRASEIRA (DESCRIÇÃO COMPLETA) -->
            <div class="card-face card-back">
              <div>
                <div class="prod-top-badges">
                  <button class="cat-tag-clickable" onclick="flipCardToFront(${pid})" title="Clique na categoria para voltar ao produto">
                    ↩ ${escapeHtml(p.category_name || "Instrumento")}
                  </button>
                  <span class="btn-desc-flip" onclick="flipCardToFront(${pid})" style="cursor:pointer;" title="Clique para voltar">
                    Descrição ✕
                  </span>
                  <div style="display:flex; gap:4px; align-items:center;">
                    ${promoBadge}
                    ${statusBadge}
                  </div>
                </div>
                <div class="card-back-body">
                  <div class="card-back-title" title="${escapeHtml(p.name)}">${escapeHtml(p.name)}</div>
                  <div class="card-back-meta">
                    ${p.brand ? `<span class="meta-pill">🏷️ ${escapeHtml(p.brand)}</span>` : ""}
                    ${p.strings_type ? `<span class="meta-pill">🎸 ${escapeHtml(p.strings_type)}</span>` : ""}
                    ${p.is_electroacoustic ? `<span class="meta-pill">⚡ Eletroacústico</span>` : ""}
                  </div>
                  <div class="card-back-desc-box">
                    <div class="card-back-desc-heading">Descrição Completa:</div>
                    <div class="card-back-desc-content">${escapeHtml(p.description || "Descrição completa não cadastrada no catálogo.")}</div>
                  </div>
                </div>
              </div>
              <div class="card-back-footer">
                <div class="card-back-price-box">
                  <div>
                    <span class="card-back-price">${formatBRL(p.effective_price_brl)}</span>
                    <span class="card-back-pix"> (⚡ ${formatBRL(p.pix_price_brl)} PIX)</span>
                  </div>
                </div>
                <button class="btn-return-card" onclick="flipCardToFront(${pid})">
                  ↩ Voltar ao Produto
                </button>
              </div>
            </div>
          </div>
        `;

        if (inStock) {
          const addBtn = card.querySelector(`#btn_add_${pid}`);
          addBtn.addEventListener("click", () => {
            const qtyEl = document.getElementById(`qty_${pid}`);
            const qty = parseInt(qtyEl ? qtyEl.textContent : "1", 10) || 1;
            addToCart(p, qty, addBtn);
          });
        }

        track.appendChild(card);
      });

      section.appendChild(track);
      return section;
    }

    function flipCardToBack(pid) {
      const cardInner = document.getElementById(`card_inner_${pid}`);
      if (cardInner) {
        cardInner.classList.add("is-flipped");
      }
    }

    function flipCardToFront(pid) {
      const cardInner = document.getElementById(`card_inner_${pid}`);
      if (cardInner) {
        cardInner.classList.remove("is-flipped");
      }
    }

    function changeCardQty(pid, delta, maxStock) {
      const el = document.getElementById(`qty_${pid}`);
      if (!el) return;
      let current = parseInt(el.textContent, 10) || 1;
      current = Math.max(1, Math.min(maxStock || 1, current + delta));
      el.textContent = String(current);
    }

    function addToCart(product, qty, btnEl) {
      const existing = cart.find(item => item.product.product_id === product.product_id);
      if (existing) {
        existing.qty = Math.min(product.stock_quantity, existing.qty + qty);
      } else {
        cart.push({ product, qty });
      }
      updateCartUI();

      if (btnEl) {
        const origText = btnEl.innerHTML;
        btnEl.innerHTML = "✔ Adicionado!";
        btnEl.style.background = "#10b981";
        setTimeout(() => {
          btnEl.innerHTML = origText;
          btnEl.style.background = "";
        }, 1000);
      }
    }

    function removeFromCart(productId) {
      cart = cart.filter(item => item.product.product_id !== productId);
      updateCartUI();
    }

    function toggleCart() {
      const drawer = document.getElementById("cartDrawer");
      drawer.classList.toggle("open");
    }

    function updateCartUI() {
      const countEl = document.getElementById("cartCount");
      const listEl = document.getElementById("cartItemsList");
      const footerEl = document.getElementById("cartFooter");
      const regTotalEl = document.getElementById("cartTotalRegular");
      const pixTotalEl = document.getElementById("cartTotalPix");

      const totalItems = cart.reduce((acc, item) => acc + item.qty, 0);
      countEl.textContent = String(totalItems);

      if (cart.length === 0) {
        listEl.innerHTML = '<p style="color:#94a3b8; font-size:0.82rem; text-align:center; padding:16px 0;">Seu carrinho está vazio.</p>';
        footerEl.style.display = "none";
        return;
      }

      footerEl.style.display = "flex";
      let sumRegular = 0;
      let sumPix = 0;

      listEl.innerHTML = cart.map(item => {
        const p = item.product;
        const subReg = p.effective_price_brl * item.qty;
        const subPix = p.pix_price_brl * item.qty;
        sumRegular += subReg;
        sumPix += subPix;
        return `
          <div class="cart-item">
            <div class="cart-item-info">
              <div class="cart-item-title">${item.qty}x ${escapeHtml(p.name)}</div>
              <div class="cart-item-sub">${formatBRL(subReg)} (PIX: ${formatBRL(subPix)})</div>
            </div>
            <button class="btn-rm" onclick="removeFromCart(${p.product_id})">✕</button>
          </div>
        `;
      }).join("");

      regTotalEl.textContent = formatBRL(sumRegular);
      pixTotalEl.textContent = formatBRL(sumPix);
    }

    function simulateCartWithAgent() {
      if (cart.length === 0) return;
      const summary = cart.map(i => `${i.qty}x ${i.product.name}`).join(", ");
      document.getElementById("cartDrawer").classList.remove("open");
      sendQuick(`Selecionei no carrinho: ${summary}. Como ficam as opções de pagamento (PIX e parcelamento sem juros) e o frete para Campo Grande?`);
    }

    function appendMsg(text, role, meta = null) {
      const wrapper = document.createElement("div");
      wrapper.className = "msg-wrapper";

      const div = document.createElement("div");
      div.className = "msg " + role;

      if (role === "bot") {
        const mdDiv = document.createElement("div");
        mdDiv.className = "md-content";
        mdDiv.innerHTML = renderMarkdown(text);
        div.appendChild(mdDiv);
      } else {
        div.textContent = text;
      }

      if (meta) {
        const badge = document.createElement("div");
        badge.className = "meta-badge";
        const tools = (meta.tools_called || []).map(t => t.tool).join(", ") || "direct";
        badge.textContent = `Model: ${meta.model_used} | Lang: ${meta.language} | Tools: ${tools}`;
        div.appendChild(badge);
      }

      wrapper.appendChild(div);

      if (role === "bot" && meta && Array.isArray(meta.products) && meta.products.length > 0) {
        const carouselEl = buildProductCarousel(meta.products);
        if (carouselEl) wrapper.appendChild(carouselEl);
      }

      messagesEl.appendChild(wrapper);
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
      cart = [];
      updateCartUI();
      messagesEl.innerHTML = '<div class="msg-wrapper"><div class="msg bot"><div class="md-content">Sessão e carrinho reiniciados! Como posso ajudar você hoje na Empório da Música? 🎶</div></div></div>';
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
            "Hybrid Architecture combining LLM Function Calling (Qwen3 primary + Fallbacks), "
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
        try:
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
                products=result.get("products", []),
                logged_failed_intent=result.get("logged_failed_intent", False),
            )
        except Exception as exc:
            return ChatResponse(
                session_id=payload.session_id,
                reply=(
                    "Tivemos uma pequena oscilação temporária de conexão com o provedor de IA. "
                    "Por favor, tente novamente em alguns segundos ou contate nosso WhatsApp (67) 3321-4500! 🎵"
                ),
                language="pt",
                confidence=0.5,
                model_used="api_error_recovery",
                tools_called=[],
                products=[],
                logged_failed_intent=True,
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
