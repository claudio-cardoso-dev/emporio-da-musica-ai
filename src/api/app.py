"""FastAPI application with OpenAPI/Swagger documentation and an embedded glassmorphism Web Chat UI with Interactive Product Carousel & Cart."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.services.cep_service import CepService
from src.services.chat_service import ChatService
from src.services.pix_service import PixService


class ChatRequest(BaseModel):
    """Request payload for sending a message to the Empório da Música AI Agent."""

    message: str = Field(..., description="Customer message in any language (PT, EN, ES).")
    session_id: str = Field(
        default="web_session_01",
        description="Unique session identifier for multi-turn conversation memory.",
    )
    cart: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Current shopping cart items from frontend for synchronization.",
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
    active_agent: Optional[str] = Field(
        default="catalog",
        description="Specialist agent that handled the conversation turn (catalog, checkout, customer, support).",
    )
    action: Optional[str] = Field(
        default=None,
        description="UI action triggered in chat (e.g. 'SHOW_REGISTER_FORM', 'SHOW_ADDRESS_FORM').",
    )
    product: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Product added to cart via chat text.",
    )
    qty: Optional[int] = Field(
        default=None,
        description="Quantity of product added to cart via chat text.",
    )
    options: List[str] = Field(
        default_factory=list,
        description="Quick interactive response buttons rendered under the message.",
    )
    prefill: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional customer data to pre-fill in inline forms.",
    )
    logged_failed_intent: bool


class ResetSessionRequest(BaseModel):
    """Payload to clear conversation history for a session."""

    session_id: str = Field(default="web_session_01")


class CustomerRegisterRequest(BaseModel):
    """Payload for registering a new customer into the store."""

    name: str = Field(..., description="Customer full name.")
    phone: Optional[str] = Field(None, description="Phone number or WhatsApp (e.g. '(67) 99999-8888').")
    email: Optional[str] = Field(None, description="Customer email address.")
    cep: Optional[str] = Field(None, description="8-digit Brazilian postal code (e.g. '79202-333').")
    address: Optional[str] = Field(None, description="Delivery street address and house/apt number.")
    city: Optional[str] = Field(None, description="City name (defaults to 'Campo Grande' or resolved via CEP).")


class CustomerAddressRequest(BaseModel):
    """Payload for updating delivery address of an existing customer."""

    customer_identifier: str = Field(
        ...,
        description="Customer primary key (phone, email, or customer_id).",
    )
    address: str = Field(..., description="Updated street address, number, and complement.")
    cep: Optional[str] = Field(None, description="8-digit Brazilian postal code.")
    city: Optional[str] = Field(None, description="City name (optional, resolved via CEP if empty).")


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
    .md-content code {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(56, 189, 248, 0.35);
      padding: 3px 6px;
      border-radius: 6px;
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 0.82rem;
      color: #38bdf8;
      word-break: break-all;
      display: inline-block;
      margin: 4px 0;
      user-select: all;
    }
    .qr-code-box {
      margin: 12px 0;
      display: block;
    }

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
      min-width: 215px;
      max-width: 215px;
      height: 236px;
      perspective: 1000px;
      position: relative;
      background: transparent;
      border: none;
      padding: 0;
      flex-shrink: 0;
      transition: transform 0.2s ease;
    }
    .prod-card:hover {
      transform: translateY(-3px);
    }
    .card-inner {
      position: relative;
      width: 100%;
      height: 100%;
      transform-style: preserve-3d;
      transition: transform 0.65s cubic-bezier(0.4, 0, 0.2, 1);
      border-radius: 12px;
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
      border-radius: 12px;
      padding: 8px 10px;
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      border: 1px solid rgba(255, 255, 255, 0.12);
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
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
      display: flex; justify-content: space-between; align-items: center; gap: 4px;
      width: 100%; margin-bottom: 2px;
    }
    .cat-tag {
      font-size: 0.64rem; font-weight: 600; color: #94a3b8; background: rgba(255,255,255,0.07);
      padding: 1px 4px; border-radius: 4px; white-space: nowrap;
      border: 1px solid transparent; flex-shrink: 0;
    }
    .cat-tag-clickable {
      font-size: 0.64rem; font-weight: 700; color: #38bdf8; background: rgba(56, 189, 248, 0.14);
      border: 1px solid rgba(56, 189, 248, 0.45); padding: 1px 4px; border-radius: 4px;
      cursor: pointer; display: inline-flex; align-items: center; gap: 3px; transition: all 0.2s ease;
      font-family: inherit; white-space: nowrap; flex-shrink: 0;
    }
    .cat-tag-clickable:hover {
      background: rgba(56, 189, 248, 0.3); border-color: #38bdf8; color: #fff; transform: scale(1.04);
    }
    .btn-desc-flip {
      font-size: 0.62rem; font-weight: 700; color: #fcd34d; background: rgba(245, 158, 11, 0.16);
      border: 1px solid rgba(245, 158, 11, 0.4); padding: 1px 4px; border-radius: 4px;
      cursor: pointer; display: inline-flex; align-items: center; gap: 2px; transition: all 0.2s ease;
      white-space: nowrap; font-family: inherit; flex-shrink: 0;
    }
    .btn-desc-flip:hover {
      background: rgba(245, 158, 11, 0.32); border-color: var(--accent); color: #fff; transform: scale(1.04);
    }
    .card-back-body {
      flex: 1; display: flex; flex-direction: column; gap: 3px; margin-top: 2px; overflow: hidden;
    }
    .card-back-title {
      font-size: 0.76rem; font-weight: 700; color: #fcd34d; line-height: 1.2;
      display: -webkit-box; -webkit-line-clamp: 1; -webkit-box-orient: vertical; overflow: hidden;
    }
    .card-back-meta {
      display: flex; flex-wrap: wrap; gap: 3px;
    }
    .meta-pill {
      font-size: 0.58rem; background: rgba(255, 255, 255, 0.07); color: #cbd5e1;
      padding: 1px 3px; border-radius: 3px; border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .card-back-desc-box {
      flex: 1; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px; padding: 4px 6px; overflow-y: auto;
      scrollbar-width: thin; scrollbar-color: rgba(245, 158, 11, 0.4) transparent;
      margin-top: 2px; max-height: 85px;
    }
    .card-back-desc-box::-webkit-scrollbar { width: 3px; }
    .card-back-desc-box::-webkit-scrollbar-thumb { background: rgba(245, 158, 11, 0.4); border-radius: 999px; }
    .card-back-desc-heading {
      font-size: 0.58rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;
      letter-spacing: 0.04em; margin-bottom: 1px;
    }
    .card-back-desc-content {
      font-size: 0.66rem; color: #e2e8f0; line-height: 1.3; white-space: pre-wrap;
    }
    .card-back-footer {
      display: flex; flex-direction: column; gap: 3px; margin-top: 3px; padding-top: 3px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .card-back-price-box {
      display: flex; justify-content: space-between; align-items: baseline;
    }
    .card-back-price {
      font-size: 0.82rem; font-weight: 800; color: #fcd34d;
    }
    .card-back-pix {
      font-size: 0.65rem; font-weight: 600; color: #6ee7b7;
    }
    .btn-return-card {
      width: 100%; padding: 5px 6px; border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.15); background: rgba(255, 255, 255, 0.08);
      color: #fff; font-weight: 700; font-size: 0.70rem; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 4px;
      transition: 0.2s; font-family: inherit;
    }
    .btn-return-card:hover {
      background: rgba(255, 255, 255, 0.18); border-color: #38bdf8; color: #38bdf8;
    }
    .promo-tag {
      font-size: 0.60rem; font-weight: 700; background: rgba(239, 68, 68, 0.2);
      color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4);
      padding: 1px 4px; border-radius: 999px;
    }
    .stock-tag {
      font-size: 0.60rem; font-weight: 600; padding: 1px 4px; border-radius: 999px;
    }
    .stock-ok { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; }
    .stock-out { background: rgba(239, 68, 68, 0.18); color: #fca5a5; }
    .prod-icon-box {
      height: 32px; border-radius: 8px; margin: 2px 0;
      background: radial-gradient(circle, rgba(245, 158, 11, 0.16), rgba(15, 23, 42, 0.6));
      display: flex; align-items: center; justify-content: center;
      font-size: 18px; border: 1px solid rgba(255,255,255,0.05);
    }
    .prod-title {
      font-size: 0.78rem; font-weight: 700; color: #fff; line-height: 1.2;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
      min-height: 28px;
    }
    .prod-desc {
      font-size: 0.72rem; color: var(--text-muted); line-height: 1.3;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .price-box {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 6px;
      padding: 4px 6px;
    }
    .old-price {
      font-size: 0.64rem; color: #94a3b8; text-decoration: line-through;
    }
    .main-price {
      font-size: 0.88rem; font-weight: 800; color: #fcd34d;
    }
    .pix-price {
      font-size: 0.66rem; color: #6ee7b7; font-weight: 600; margin-top: 1px;
    }
    .card-actions {
      display: flex; flex-direction: column; gap: 3px; margin-top: 3px;
    }
    .qty-row {
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border);
      border-radius: 6px; padding: 1px 4px;
    }
    .qty-btn {
      width: 20px; height: 20px; border-radius: 4px; border: none;
      background: rgba(255,255,255,0.08); color: #fff; font-weight: 700;
      cursor: pointer; transition: 0.15s; font-size: 0.75rem;
    }
    .qty-btn:hover:not(:disabled) { background: var(--accent); color: #0f172a; }
    .qty-val { font-size: 0.75rem; font-weight: 700; color: #fff; min-width: 16px; text-align: center; }
    .btn-add-cart {
      width: 100%; padding: 5px 6px; border-radius: 6px; border: none;
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #0f172a; font-weight: 800; font-size: 0.72rem; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 4px;
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

    /* Customer Registration & Address Modal */
    .modal-backdrop {
      position: absolute; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(10, 15, 29, 0.82); backdrop-filter: blur(8px);
      z-index: 60; display: none; align-items: center; justify-content: center;
      padding: 16px;
    }
    .modal-backdrop.open { display: flex; }
    .customer-modal {
      width: 100%; max-width: 520px; max-height: 90vh;
      background: linear-gradient(165deg, rgba(20, 30, 48, 0.98), rgba(15, 23, 42, 0.98));
      border: 1px solid rgba(245, 158, 11, 0.45); border-radius: 18px;
      box-shadow: 0 25px 60px rgba(0, 0, 0, 0.8);
      display: flex; flex-direction: column; overflow: hidden;
    }
    .modal-header {
      padding: 16px 20px; border-bottom: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: center;
    }
    .modal-title { font-size: 1rem; font-weight: 700; color: #fcd34d; display: flex; align-items: center; gap: 8px; }
    .btn-close-modal {
      background: transparent; border: none; color: #94a3b8; font-size: 1.2rem; cursor: pointer;
      padding: 4px 8px; border-radius: 6px; line-height: 1;
    }
    .btn-close-modal:hover { color: #fff; background: rgba(255,255,255,0.1); }
    .modal-tabs {
      display: flex; border-bottom: 1px solid var(--border); background: rgba(15, 23, 42, 0.6);
    }
    .modal-tab-btn {
      flex: 1; padding: 11px 16px; background: transparent; border: none;
      color: #94a3b8; font-weight: 600; font-size: 0.82rem; cursor: pointer;
      border-bottom: 2px solid transparent; transition: all 0.2s; font-family: inherit;
    }
    .modal-tab-btn.active {
      color: #fcd34d; border-bottom-color: var(--accent); background: rgba(245, 158, 11, 0.08);
    }
    .modal-body {
      padding: 18px 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px;
    }
    .form-group { display: flex; flex-direction: column; gap: 4px; }
    .form-label { font-size: 0.76rem; font-weight: 600; color: #cbd5e1; }
    .form-row { display: flex; gap: 10px; }
    .form-input {
      width: 100%; background: rgba(15, 23, 42, 0.85); border: 1px solid var(--border);
      border-radius: 10px; padding: 9px 12px; color: #fff; font-size: 0.88rem; outline: none;
      font-family: inherit; transition: 0.2s;
    }
    .form-input:focus { border-color: var(--accent); }
    .cep-row { display: flex; gap: 8px; align-items: center; }
    .btn-search-cep {
      background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.4);
      color: #fcd34d; padding: 9px 14px; border-radius: 10px; font-weight: 700;
      font-size: 0.8rem; cursor: pointer; white-space: nowrap; font-family: inherit; transition: 0.2s;
    }
    .btn-search-cep:hover { background: rgba(245, 158, 11, 0.35); }
    .cep-feedback { font-size: 0.74rem; color: #6ee7b7; margin-top: 2px; }
    .modal-footer {
      padding: 14px 20px; border-top: 1px solid var(--border); background: rgba(15, 23, 42, 0.6);
      display: flex; justify-content: flex-end; gap: 10px;
    }
    .btn-submit-modal {
      background: var(--accent); color: #0f172a; font-weight: 800; border: none;
      padding: 10px 18px; border-radius: 10px; cursor: pointer; font-size: 0.86rem;
      transition: 0.2s; font-family: inherit;
    }
    .btn-submit-modal:hover { background: var(--accent-hover); color: #fff; }
    .alert-banner {
      padding: 8px 12px; border-radius: 8px; font-size: 0.8rem; display: none; margin-bottom: 6px;
    }
    .alert-banner.success { background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #6ee7b7; display: block; }
    .alert-banner.error { background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #fca5a5; display: block; }

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

    /* Inline Interactive Quick Reply Options */
    .chat-options-row {
      display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;
    }
    .chat-option-btn {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(234, 88, 12, 0.2));
      border: 1px solid rgba(245, 158, 11, 0.55); color: #fcd34d; font-weight: 700;
      padding: 8px 14px; border-radius: 10px; font-size: 0.82rem; cursor: pointer;
      font-family: inherit; transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px;
    }
    .chat-option-btn:hover {
      background: linear-gradient(135deg, #f59e0b, #ea580c); color: #0f172a; transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(245, 158, 11, 0.35);
    }

    /* Inline Chat Forms (Cadastro e Endereço) */
    .chat-form-card {
      margin-top: 12px; background: rgba(15, 23, 42, 0.95);
      border: 1px solid rgba(245, 158, 11, 0.45); border-radius: 14px;
      padding: 14px 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.45);
      animation: fadeInForm 0.3s ease-out;
    }
    @keyframes fadeInForm {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .chat-form-header {
      display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
      padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .chat-form-icon { font-size: 1.4rem; }
    .chat-form-title { font-size: 0.92rem; font-weight: 700; color: #fcd34d; }
    .chat-form-subtitle { font-size: 0.74rem; color: #94a3b8; }
    .chat-form-body { display: flex; flex-direction: column; gap: 10px; }
    .btn-chat-form-submit {
      margin-top: 6px; background: var(--accent); color: #0f172a;
      font-weight: 800; border: none; padding: 10px 16px; border-radius: 10px;
      cursor: pointer; font-size: 0.84rem; font-family: inherit; transition: 0.2s;
    }
    .btn-chat-form-submit:hover { background: var(--accent-hover); color: #fff; }
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

    <!-- Customer Registration & Address Modal (Busca CEP) -->
    <div class="modal-backdrop" id="customerModalBackdrop" onclick="if(event.target===this) closeCustomerModal()">
      <div class="customer-modal">
        <div class="modal-header">
          <div class="modal-title"><span>👤</span> Cadastro &amp; Endereço (ViaCEP)</div>
          <button class="btn-close-modal" onclick="closeCustomerModal()">✕</button>
        </div>
        <div class="modal-tabs">
          <button class="modal-tab-btn active" id="tabBtnRegister" onclick="switchCustomerTab('register')">Novo Cadastro</button>
          <button class="modal-tab-btn" id="tabBtnAddress" onclick="switchCustomerTab('address')">Atualizar Endereço</button>
        </div>
        <div class="modal-body">
          <div id="modalAlert" class="alert-banner"></div>

          <!-- ABA 1: NOVO CADASTRO -->
          <div id="tabContentRegister">
            <div class="form-group">
              <label class="form-label">Nome Completo *</label>
              <input type="text" id="regName" class="form-input" placeholder="Ex: Lucas Gabriel Silveira" />
            </div>
            <div class="form-row" style="margin-top:8px;">
              <div class="form-group" style="flex:1;">
                <label class="form-label">Telefone / WhatsApp *</label>
                <input type="text" id="regPhone" class="form-input" placeholder="(67) 99123-4567" />
              </div>
              <div class="form-group" style="flex:1;">
                <label class="form-label">E-mail *</label>
                <input type="email" id="regEmail" class="form-input" placeholder="lucas@exemplo.com" />
              </div>
            </div>
            <div class="form-group" style="margin-top:8px;">
              <label class="form-label">CEP (Busca Automática ViaCEP) *</label>
              <div class="cep-row">
                <input type="text" id="regCep" class="form-input" placeholder="79202-333" maxlength="9" oninput="handleCepInput('register')" />
                <button type="button" class="btn-search-cep" onclick="triggerCepSearch('register')">🔍 Buscar CEP</button>
              </div>
              <div id="regCepFeedback" class="cep-feedback"></div>
            </div>
            <div class="form-group" style="margin-top:8px;">
              <label class="form-label">Logradouro / Rua</label>
              <input type="text" id="regAddress" class="form-input" placeholder="Rua / Av..." />
            </div>
            <div class="form-row" style="margin-top:8px;">
              <div class="form-group" style="flex:1;">
                <label class="form-label">Número e Complemento</label>
                <input type="text" id="regNumber" class="form-input" placeholder="123, Apto 402" />
              </div>
              <div class="form-group" style="flex:1;">
                <label class="form-label">Bairro</label>
                <input type="text" id="regNeighborhood" class="form-input" placeholder="Centro" />
              </div>
            </div>
            <div class="form-row" style="margin-top:8px;">
              <div class="form-group" style="flex:2;">
                <label class="form-label">Cidade</label>
                <input type="text" id="regCity" class="form-input" value="Campo Grande" />
              </div>
              <div class="form-group" style="flex:1;">
                <label class="form-label">UF</label>
                <input type="text" id="regState" class="form-input" value="MS" maxlength="2" />
              </div>
            </div>
          </div>

          <!-- ABA 2: ATUALIZAR ENDEREÇO -->
          <div id="tabContentAddress" style="display:none;">
            <div class="form-group">
              <label class="form-label">Chave de Identificação (Telefone ou E-mail cadastrado) *</label>
              <input type="text" id="updIdentifier" class="form-input" placeholder="(67) 99123-4567 ou email@exemplo.com" />
            </div>
            <div class="form-group" style="margin-top:8px;">
              <label class="form-label">Novo CEP (Busca Automática ViaCEP)</label>
              <div class="cep-row">
                <input type="text" id="updCep" class="form-input" placeholder="79202-333" maxlength="9" oninput="handleCepInput('address')" />
                <button type="button" class="btn-search-cep" onclick="triggerCepSearch('address')">🔍 Buscar CEP</button>
              </div>
              <div id="updCepFeedback" class="cep-feedback"></div>
            </div>
            <div class="form-group" style="margin-top:8px;">
              <label class="form-label">Novo Logradouro / Rua *</label>
              <input type="text" id="updAddress" class="form-input" placeholder="Rua / Av..." />
            </div>
            <div class="form-row" style="margin-top:8px;">
              <div class="form-group" style="flex:1;">
                <label class="form-label">Número e Complemento</label>
                <input type="text" id="updNumber" class="form-input" placeholder="456, Bloco B" />
              </div>
              <div class="form-group" style="flex:1;">
                <label class="form-label">Bairro</label>
                <input type="text" id="updNeighborhood" class="form-input" placeholder="Bairro" />
              </div>
            </div>
            <div class="form-row" style="margin-top:8px;">
              <div class="form-group" style="flex:2;">
                <label class="form-label">Cidade</label>
                <input type="text" id="updCity" class="form-input" value="Campo Grande" />
              </div>
              <div class="form-group" style="flex:1;">
                <label class="form-label">UF</label>
                <input type="text" id="updState" class="form-input" value="MS" maxlength="2" />
              </div>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-ghost" onclick="closeCustomerModal()">Cancelar</button>
          <button class="btn-submit-modal" id="btnSubmitModal" onclick="submitCustomerForm()">Salvar Cadastro</button>
        </div>
      </div>
    </div>

    <div class="messages" id="messages">
      <div class="msg-wrapper">
        <div class="msg bot">
          <div class="md-content">Olá! Seja muito bem-vindo(a) à <strong>Empório da Música</strong>! 🎸🎶<br/><br/>Sou seu assistente virtual. Para começarmos o seu atendimento e identificarmos seu cadastro, por favor informe o seu <strong>número de telefone/WhatsApp</strong> ou <strong>e-mail</strong>.</div>
        </div>
      </div>
    </div>

    <div class="chips" id="chipsContainer" style="display:none;"></div>

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
      // Markdown Images: ![alt](url)
      s = s.replace(/!\\[([^\\]]*)\\]\\(([^)]+)\\)/g, '<div class="qr-code-box"><img src="$2" alt="$1" style="max-width:230px; width:100%; height:auto; border-radius:12px; background:#ffffff; padding:10px; box-shadow:0 8px 24px rgba(0,0,0,0.45); border:1px solid rgba(255,255,255,0.2); display:block;" /></div>');
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
        if (line.startsWith("|") && i + 1 < lines.length && /^[|\\s:-]+$/.test(lines[i + 1].trim())) {
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

      const totalItems = cart.reduce((acc, item) => acc + item.qty, 0);
      const totalRegular = cart.reduce((acc, i) => acc + (i.product.effective_price_brl || i.product.price_brl) * i.qty, 0);
      const itemLabel = totalItems === 1 ? "1 item" : `${totalItems} itens`;
      appendMsg(
        `🛒 **${qty}x ${escapeHtml(product.name)}** adicionado ao seu carrinho!\n\n` +
        `Seu carrinho agora possui **${itemLabel}** (Total: **${formatBRL(totalRegular)}**).\n\n` +
        `Deseja adicionar mais algum instrumento ou ver mais opções?`,
        "bot",
        {
          active_agent: "checkout",
          options: ["🎸 Ver mais produtos", "🛒 Ver Carrinho", "⚡ Fechar Pedido"]
        }
      );
    }

    function removeFromCart(productId) {
      const found = cart.find(item => item.product.product_id === productId);
      const pName = found ? found.product.name : "Item";
      cart = cart.filter(item => item.product.product_id !== productId);
      updateCartUI();
      appendMsg(`🗑️ **${pName}** foi removido do seu carrinho.`, "bot", {
        active_agent: "checkout",
        options: cart.length > 0 ? ["🛒 Ver Carrinho", "⚡ Fechar Pedido", "🎸 Continuar Comprando"] : ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"]
      });
    }

    function toggleCart() {
      showCartInChat();
    }

    function updateCartUI() {
      const countEl = document.getElementById("cartCount");
      const listEl = document.getElementById("cartItemsList");
      const footerEl = document.getElementById("cartFooter");
      const regTotalEl = document.getElementById("cartTotalRegular");
      const pixTotalEl = document.getElementById("cartTotalPix");

      const totalItems = cart.reduce((acc, item) => acc + item.qty, 0);
      if (countEl) countEl.textContent = String(totalItems);

      if (!listEl || !footerEl) return;

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

      if (regTotalEl) regTotalEl.textContent = formatBRL(sumRegular);
      if (pixTotalEl) pixTotalEl.textContent = formatBRL(sumPix);
    }

    function showCartInChat() {
      if (!cart || cart.length === 0) {
        appendMsg(
          `🛒 **Seu carrinho está vazio no momento.**\n\nVocê ainda não adicionou nenhum produto. Que tal conferir os instrumentos disponíveis no catálogo?`,
          "bot",
          {
            active_agent: "checkout",
            options: ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"]
          }
        );
        return;
      }

      let totalRegular = 0;
      let totalPix = 0;
      let mdRows = "";

      cart.forEach(item => {
        const p = item.product;
        const uPrice = p.effective_price_brl || p.price_brl;
        const sub = uPrice * item.qty;
        const pPix = p.pix_price_brl || (uPrice * 0.95);
        totalRegular += sub;
        totalPix += (pPix * item.qty);
        mdRows += `| ${p.name} | ${item.qty} | ${formatBRL(uPrice)} | ${formatBRL(sub)} |\n`;
      });

      const totalItems = cart.reduce((acc, item) => acc + item.qty, 0);
      const itemWord = totalItems === 1 ? "item" : "itens";

      // Maximum installments based on store policy:
      // 1x-3x: min R$50 | 4x-6x: min R$80 | 7x-12x: min R$100
      let maxInst = 1;
      for (let n = 12; n >= 1; n--) {
        const val = totalRegular / n;
        if (n >= 7 && val >= 100.0) { maxInst = n; break; }
        if (n >= 4 && n <= 6 && val >= 80.0) { maxInst = n; break; }
        if (n >= 2 && n <= 3 && val >= 50.0) { maxInst = n; break; }
        if (n === 1) { maxInst = 1; break; }
      }

      const instVal = totalRegular / maxInst;
      const instText = maxInst > 1 
        ? `em até **${maxInst}x de ${formatBRL(instVal)} sem juros**` 
        : `1x de ${formatBRL(totalRegular)}`;

      const freeShipping = totalRegular >= 300;
      const shippingText = freeShipping 
        ? "• 🚚 **Frete Grátis** para Campo Grande/MS via motoboy próprio da loja (1 a 3 dias úteis)."
        : "• 🚚 **Frete para Campo Grande/MS:** R$ 25,00 via motoboy próprio (1 a 3 dias úteis).";

      const combinedText = totalRegular >= 2000.0 
        ? `• 🌟 **Pagamento Combinado (PIX + Cartão):** Elegível para este pedido (valor total acima de R$ 2.000,00)!\n` 
        : "";

      const cartMd = `### 🛒 Itens no seu Carrinho (${totalItems} ${itemWord})\n\n` +
        `| Item | Quantidade | Valor Unitário | Subtotal |\n` +
        `| :--- | :---: | :---: | :---: |\n` +
        mdRows +
        `| **Total do Pedido** | **${totalItems} ${itemWord}** | | **${formatBRL(totalRegular)}** |\n\n` +
        `### 💳 Opções de Pagamento:\n` +
        `• ⚡ **À vista no PIX:** **${formatBRL(totalPix)}** (5% de desconto à vista)\n` +
        `• 💳 **Cartão de Crédito:** ${instText}\n` +
        combinedText +
        `\n### 📦 Entrega:\n` +
        `${shippingText}\n\n` +
        `--- \n` +
        `Deseja fechar o seu pedido agora ou adicionar mais produtos?`;

      appendMsg(cartMd, "bot", {
        active_agent: "checkout",
        options: ["⚡ Fechar Pedido", "🗑️ Esvaziar Carrinho", "🎸 Continuar Comprando"]
      });
    }

    function simulateCartWithAgent() {
      if (cart.length === 0) {
        showCartInChat();
        return;
      }
      const totalRegular = cart.reduce((acc, i) => acc + (i.product.effective_price_brl || i.product.price_brl) * i.qty, 0);
      const itemsList = cart.map(i => `${i.qty}x ${i.product.name} (R$ ${(i.product.effective_price_brl || i.product.price_brl).toFixed(2).replace('.', ',')})`).join(", ");
      const totalFormatted = totalRegular.toFixed(2).replace('.', ',');
      appendMsg(`Selecionei no carrinho: ${itemsList}. Valor total: R$ ${totalFormatted}. Quero fechar meu pedido!`, "user");
      sendMessageDirect(`Selecionei no carrinho: ${itemsList}. Valor total: R$ ${totalFormatted}. Quero fechar meu pedido!`, false);
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

        // 1. Interactive quick options (e.g. "Sim, sou novo cliente")
        if (meta && Array.isArray(meta.options) && meta.options.length > 0) {
          const optRow = document.createElement("div");
          optRow.className = "chat-options-row";
          meta.options.forEach(optText => {
            const btn = document.createElement("button");
            btn.className = "chat-option-btn";
            btn.textContent = optText;
            btn.onclick = () => {
              optRow.style.display = "none";
              const clean = optText.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9\\s]/g, '').trim();
              if (clean.includes("fechar pedido") || clean.includes("finalizar compra") || clean.includes("fechar carrinho") || clean.includes("concluir pedido")) {
                simulateCartWithAgent();
              } else if (clean.includes("ver carrinho")) {
                showCartInChat();
              } else if (clean.includes("continuar comprando") || clean.includes("ver mais produtos")) {
                sendQuick("Quero ver mais instrumentos musicais");
              } else {
                sendQuick(optText);
              }
            };
            optRow.appendChild(btn);
          });
          div.appendChild(optRow);
        }

        // 2. Inline Registration Form (Step 1)
        if (meta && meta.action === "SHOW_REGISTER_FORM") {
          const formCard = document.createElement("div");
          formCard.className = "chat-form-card";
          formCard.id = "chatRegisterForm";
          const prefillEmail = (meta.prefill && meta.prefill.email) || window.lastUserEmail || "";
          const prefillPhone = (meta.prefill && meta.prefill.phone) || window.lastUserPhone || "";
          formCard.innerHTML = `
            <div class="chat-form-header">
              <span class="chat-form-icon">👤</span>
              <div>
                <div class="chat-form-title">Cadastro de Novo Cliente</div>
                <div class="chat-form-subtitle">Preencha seus dados de contato para prosseguirmos</div>
              </div>
            </div>
            <div class="chat-form-body">
              <div class="form-group">
                <label class="form-label">Nome Completo *</label>
                <input type="text" class="form-input" id="chatRegName" placeholder="Ex: Carlos Eduardo Silva" />
              </div>
              <div class="form-row">
                <div class="form-group" style="flex:1;">
                  <label class="form-label">Telefone / WhatsApp *</label>
                  <input type="tel" class="form-input" id="chatRegPhone" value="${prefillPhone}" placeholder="(67) 99999-8888" />
                </div>
                <div class="form-group" style="flex:1;">
                  <label class="form-label">E-mail</label>
                  <input type="email" class="form-input" id="chatRegEmail" value="${prefillEmail}" placeholder="seu@email.com" />
                </div>
              </div>
              <div id="chatRegAlert" style="display:none; color:#fca5a5; font-size:0.75rem;"></div>
              <button class="btn-chat-form-submit" onclick="submitChatRegisterForm()">
                Avançar para Endereço ➔
              </button>
            </div>
          `;
          div.appendChild(formCard);
        }

        // 3. Inline Address Form (Step 2)
        if (meta && meta.action === "SHOW_ADDRESS_FORM") {
          const formCard = document.createElement("div");
          formCard.className = "chat-form-card";
          formCard.id = "chatAddressForm";
          formCard.innerHTML = `
            <div class="chat-form-header">
              <span class="chat-form-icon">📍</span>
              <div>
                <div class="chat-form-title">Endereço de Entrega</div>
                <div class="chat-form-subtitle">Digite seu CEP para buscar rua e bairro automaticamente</div>
              </div>
            </div>
            <div class="chat-form-body">
              <div class="form-row">
                <div class="form-group" style="flex:1;">
                  <label class="form-label">CEP (Busca CEP) *</label>
                  <input type="text" class="form-input" id="chatAddrCep" placeholder="79202-333" oninput="handleChatCepInput()" />
                </div>
                <div class="form-group" style="flex:2;">
                  <label class="form-label">Logradouro / Rua *</label>
                  <input type="text" class="form-input" id="chatAddrStreet" placeholder="Rua ou Avenida" />
                </div>
              </div>
              <div id="chatCepFeedback" class="cep-feedback"></div>
              <div class="form-row">
                <div class="form-group" style="flex:1;">
                  <label class="form-label">Número *</label>
                  <input type="text" class="form-input" id="chatAddrNumber" placeholder="123" />
                </div>
                <div class="form-group" style="flex:2;">
                  <label class="form-label">Complemento</label>
                  <input type="text" class="form-input" id="chatAddrComplement" placeholder="Apto 4B, Bloco C (opcional)" />
                </div>
              </div>
              <div class="form-row">
                <div class="form-group" style="flex:1;">
                  <label class="form-label">Bairro</label>
                  <input type="text" class="form-input" id="chatAddrNeighborhood" placeholder="Bairro" />
                </div>
                <div class="form-group" style="flex:1;">
                  <label class="form-label">Cidade / UF</label>
                  <input type="text" class="form-input" id="chatAddrCity" placeholder="Campo Grande - MS" />
                </div>
              </div>
              <div id="chatAddrAlert" style="display:none; color:#fca5a5; font-size:0.75rem;"></div>
              <button class="btn-chat-form-submit" onclick="submitChatAddressForm()">
                Finalizar Cadastro Completo ✔
              </button>
            </div>
          `;
          div.appendChild(formCard);
        }
      } else {
        div.textContent = text;
      }

      if (meta) {
        const badge = document.createElement("div");
        badge.className = "meta-badge";
        const tools = (meta.tools_called || []).map(t => t.tool).join(", ") || "direct";
        const agentTag = meta.active_agent ? `Agent: ${meta.active_agent} | ` : "";
        badge.textContent = `${agentTag}Model: ${meta.model_used} | Lang: ${meta.language} | Tools: ${tools}`;
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

    let pendingChatCustomer = { name: "", phone: "", email: "" };

    function submitChatRegisterForm() {
      const name = document.getElementById("chatRegName").value.trim();
      const phone = document.getElementById("chatRegPhone").value.trim();
      const email = document.getElementById("chatRegEmail").value.trim();
      const alertEl = document.getElementById("chatRegAlert");

      if (!name) {
        alertEl.textContent = "Por favor, informe seu nome completo.";
        alertEl.style.display = "block";
        return;
      }
      if (!phone && !email) {
        alertEl.textContent = "Informe pelo menos seu telefone ou e-mail.";
        alertEl.style.display = "block";
        return;
      }
      alertEl.style.display = "none";
      pendingChatCustomer = { name, phone, email };

      const formCard = document.getElementById("chatRegisterForm");
      if (formCard) formCard.style.display = "none";

      const summaryText = `Preenchi meus dados: Nome: ${name}, Contato: ${phone || email}.`;
      appendMsg(summaryText, "user");

      const payloadMsg = "[DADOS_DO_CADASTRO]: " + JSON.stringify(pendingChatCustomer);
      sendMessageDirect(payloadMsg);
    }

    let chatCepTimeout = null;
    function handleChatCepInput() {
      clearTimeout(chatCepTimeout);
      chatCepTimeout = setTimeout(async () => {
        const cepInput = document.getElementById("chatAddrCep");
        const feedbackEl = document.getElementById("chatCepFeedback");
        if (!cepInput || !feedbackEl) return;
        const rawCep = (cepInput.value || "").replace(/\\D/g, "");
        if (rawCep.length !== 8) {
          if (rawCep.length > 0) feedbackEl.textContent = "Digite os 8 dígitos do CEP.";
          return;
        }
        feedbackEl.textContent = "Buscando CEP no ViaCEP...";
        feedbackEl.style.color = "#fcd34d";
        try {
          const res = await fetch(`/api/cep/${rawCep}`);
          const data = await res.json();
          if (data && data.found) {
            feedbackEl.textContent = `✓ ${data.city} - ${data.state} (${data.neighborhood || "Centro"})`;
            feedbackEl.style.color = "#6ee7b7";
            if (data.street) document.getElementById("chatAddrStreet").value = data.street;
            if (data.neighborhood) document.getElementById("chatAddrNeighborhood").value = data.neighborhood;
            if (data.city) document.getElementById("chatAddrCity").value = `${data.city} - ${data.state}`;
          } else {
            feedbackEl.textContent = "CEP não localizado. Preencha o endereço manualmente.";
            feedbackEl.style.color = "#fca5a5";
          }
        } catch (e) {
          feedbackEl.textContent = "Erro ao buscar CEP. Preencha manualmente.";
          feedbackEl.style.color = "#fca5a5";
        }
      }, 500);
    }

    function submitChatAddressForm() {
      const cep = document.getElementById("chatAddrCep").value.trim();
      const street = document.getElementById("chatAddrStreet").value.trim();
      const number = document.getElementById("chatAddrNumber").value.trim();
      const complement = document.getElementById("chatAddrComplement").value.trim();
      const neighborhood = document.getElementById("chatAddrNeighborhood").value.trim();
      const city = document.getElementById("chatAddrCity").value.trim();
      const alertEl = document.getElementById("chatAddrAlert");

      if (!street) {
        alertEl.textContent = "Por favor, informe a rua ou logradouro.";
        alertEl.style.display = "block";
        return;
      }

      alertEl.style.display = "none";
      const formCard = document.getElementById("chatAddressForm");
      if (formCard) formCard.style.display = "none";

      const addrData = {
        name: pendingChatCustomer.name,
        phone: pendingChatCustomer.phone,
        email: pendingChatCustomer.email,
        cep: cep || null,
        street: street,
        number: number,
        complement: complement,
        neighborhood: neighborhood,
        city: city || "Campo Grande - MS",
      };

      const summaryText = `Preenchi meu endereço: ${street}, ${number || "s/n"} - ${neighborhood}, ${city} (CEP: ${cep || "não informado"}).`;
      appendMsg(summaryText, "user");

      const payloadMsg = "[DADOS_DO_ENDERECO]: " + JSON.stringify(addrData);
      sendMessageDirect(payloadMsg);
    }

    function sendQuick(text) {
      inputEl.value = text;
      sendMessage();
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text) return;
      inputEl.value = "";
      if (text.includes("@")) {
        const m = text.match(/[\\w\\.-]+@[\\w\\.-]+\\.\\w+/);
        if (m) window.lastUserEmail = m[0];
      }
      const digits = text.replace(/\\D/g, "");
      if (digits.length >= 8 && digits.length <= 13) {
        window.lastUserPhone = text;
      }

      const lower = text.toLowerCase().normalize("NFD").replace(/[\\u0300-\\u036f]/g, "").trim();

      // Intercept Clear Cart
      if (lower === "esvaziar carrinho" || lower === "limpar carrinho" || lower === "remover itens do carrinho" || lower === "apagar carrinho") {
        appendMsg(text, "user");
        cart = [];
        updateCartUI();
        appendMsg("🗑️ **Seu carrinho foi esvaziado.**\\n\\nComo posso te ajudar agora? Você pode explorar nossos instrumentos musicais:", "bot", {
          active_agent: "checkout",
          options: ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção"]
        });
        return;
      }

      // Intercept Cart Inquiries
      const isCartQuery = [
        "carrinho", "meu carrinho", "ver carrinho", "itens no carrinho", "itens do carrinho",
        "o que tem no carrinho", "o que tem no meu carrinho", "quais itens estao no carrinho",
        "quais produtos estao no carrinho", "mostrar carrinho", "exibir carrinho", "abrir carrinho",
        "ver meu carrinho", "como esta meu carrinho", "produtos no carrinho", "meus produtos"
      ].some(k => lower === k || lower.startsWith(k + " ") || lower.endsWith(" " + k) || lower === k + "?");

      if (isCartQuery) {
        appendMsg(text, "user");
        showCartInChat();
        return;
      }

      // Intercept Continuar Comprando
      if (lower === "continuar comprando" || lower === "ver mais produtos") {
        appendMsg(text, "user");
        appendMsg("Fique à vontade! Que tipo de instrumento você gostaria de conferir agora? 🎶", "bot", {
          active_agent: "catalog",
          options: ["🎸 Violões até R$1000", "🌴 Ver Ukuleles", "🔥 Produtos em Promoção", "📍 Endereço da Loja"]
        });
        return;
      }

      // Intercept Fechar Pedido when cart has items
      const isCheckoutIntentText = !text.startsWith("Selecionei no carrinho:") && (
        /(?:fechar|finalizar|concluir|terminar)\\s*(?:o|meu|o\\s*meu|esse|este|a)?\\s*(?:pedido|carrinho|compra)|(?:vamos|desejo)\\s*(?:fechar|finalizar|pagar)|(?:quero)\\s*(?:fechar|finalizar|pagar)(?:\\s*(?:o|meu)?\\s*(?:pedido|carrinho|compra))?|checkout/i.test(lower)
      );
      if (isCheckoutIntentText && cart.length > 0) {
        simulateCartWithAgent();
        return;
      }

      appendMsg(text, "user");
      try {
        const cartPayload = cart.map(i => ({
          product_id: i.product.product_id,
          name: i.product.name,
          price_brl: i.product.price_brl,
          effective_price_brl: i.product.effective_price_brl || i.product.price_brl,
          qty: i.qty
        }));
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: sessionId, cart: cartPayload })
        });
        const data = await res.json();
        if (data.action === "ADD_TO_CART" && data.product) {
          const p = data.product;
          const addQty = parseInt(data.qty, 10) || 1;
          const existing = cart.find(item => item.product.product_id === p.product_id);
          if (existing) {
            existing.qty = Math.min(p.stock_quantity || 99, existing.qty + addQty);
          } else {
            cart.push({ product: p, qty: addQty });
          }
          updateCartUI();
        } else if (data.action === "CLEAR_CART" || data.model_used === "direct_checkout_confirmed") {
          cart = [];
          updateCartUI();
        }
        appendMsg(data.reply, "bot", data);
      } catch (e) {
        appendMsg("Erro ao comunicar com o servidor.", "bot");
      }
    }

    function focusUserInput() {
      const el = document.getElementById("userInput");
      if (el) el.focus();
    }

    function sendStoreInfo() {
      sendQuick("Qual o endereço e horário de funcionamento da loja?");
    }

    async function resetChat() {
      await fetch("/api/session/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
      cart = [];
      updateCartUI();
      messagesEl.innerHTML = '<div class="msg-wrapper"><div class="msg bot"><div class="md-content">Olá! Seja muito bem-vindo(a) à <strong>Empório da Música</strong>! 🎸🎶<br/><br/>Sou seu assistente virtual. Para começarmos o seu atendimento e identificarmos seu cadastro, por favor informe o seu <strong>número de telefone/WhatsApp</strong> ou <strong>e-mail</strong>.</div></div></div>';
      const chipsEl = document.getElementById("chipsContainer");
      if (chipsEl) {
        chipsEl.innerHTML = "";
        chipsEl.style.display = "none";
      }
    }

    let activeCustomerTab = "register";

    function openCustomerModal() {
      document.getElementById("customerModalBackdrop").classList.add("open");
      hideModalAlert();
    }

    function closeCustomerModal() {
      document.getElementById("customerModalBackdrop").classList.remove("open");
    }

    function switchCustomerTab(tab) {
      activeCustomerTab = tab;
      const isReg = tab === "register";
      document.getElementById("tabBtnRegister").classList.toggle("active", isReg);
      document.getElementById("tabBtnAddress").classList.toggle("active", !isReg);
      document.getElementById("tabContentRegister").style.display = isReg ? "block" : "none";
      document.getElementById("tabContentAddress").style.display = isReg ? "none" : "block";
      document.getElementById("btnSubmitModal").textContent = isReg ? "Salvar Cadastro" : "Atualizar Endereço";
      hideModalAlert();
    }

    function showModalAlert(text, type) {
      const el = document.getElementById("modalAlert");
      el.className = `alert-banner ${type}`;
      el.textContent = text;
      el.style.display = "block";
    }

    function hideModalAlert() {
      const el = document.getElementById("modalAlert");
      el.style.display = "none";
      el.textContent = "";
    }

    let cepTimeout = null;
    function handleCepInput(target) {
      clearTimeout(cepTimeout);
      cepTimeout = setTimeout(() => {
        triggerCepSearch(target);
      }, 500);
    }

    async function triggerCepSearch(target) {
      const isReg = target === "register";
      const cepInput = document.getElementById(isReg ? "regCep" : "updCep");
      const feedbackEl = document.getElementById(isReg ? "regCepFeedback" : "updCepFeedback");
      const rawCep = (cepInput.value || "").replace(/\\D/g, "");
      if (rawCep.length !== 8) {
        if (rawCep.length > 0) feedbackEl.textContent = "Digite os 8 dígitos do CEP.";
        return;
      }
      feedbackEl.textContent = "Buscando CEP no ViaCEP...";
      feedbackEl.style.color = "#fcd34d";
      try {
        const res = await fetch(`/api/cep/${rawCep}`);
        const data = await res.json();
        if (data && data.found) {
          feedbackEl.textContent = `✓ ${data.city} - ${data.state} (${data.neighborhood || "Centro"})`;
          feedbackEl.style.color = "#6ee7b7";
          if (isReg) {
            if (data.street) document.getElementById("regAddress").value = data.street;
            if (data.neighborhood) document.getElementById("regNeighborhood").value = data.neighborhood;
            if (data.city) document.getElementById("regCity").value = data.city;
            if (data.state) document.getElementById("regState").value = data.state;
          } else {
            if (data.street) document.getElementById("updAddress").value = data.street;
            if (data.neighborhood) document.getElementById("updNeighborhood").value = data.neighborhood;
            if (data.city) document.getElementById("updCity").value = data.city;
            if (data.state) document.getElementById("updState").value = data.state;
          }
        } else {
          feedbackEl.textContent = "CEP não localizado ou inválido. Preencha manualmente.";
          feedbackEl.style.color = "#fca5a5";
        }
      } catch (e) {
        feedbackEl.textContent = "Erro na consulta do CEP. Preencha manualmente.";
        feedbackEl.style.color = "#fca5a5";
      }
    }

    async function submitCustomerForm() {
      hideModalAlert();
      if (activeCustomerTab === "register") {
        const name = document.getElementById("regName").value.trim();
        const phone = document.getElementById("regPhone").value.trim();
        const email = document.getElementById("regEmail").value.trim();
        const cep = document.getElementById("regCep").value.trim();
        const street = document.getElementById("regAddress").value.trim();
        const number = document.getElementById("regNumber").value.trim();
        const neighborhood = document.getElementById("regNeighborhood").value.trim();
        const city = document.getElementById("regCity").value.trim();
        const state = document.getElementById("regState").value.trim();

        if (!name) {
          showModalAlert("Informe seu nome completo.", "error");
          return;
        }
        if (!phone && !email) {
          showModalAlert("Informe pelo menos o WhatsApp/Telefone ou E-mail.", "error");
          return;
        }

        let fullAddress = street;
        if (number) fullAddress += `, ${number}`;
        if (neighborhood) fullAddress += ` - ${neighborhood}`;
        if (state) fullAddress += ` - ${state}`;

        try {
          const res = await fetch("/api/customer/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: name,
              phone: phone || null,
              email: email || null,
              cep: cep || null,
              address: fullAddress || null,
              city: city || "Campo Grande",
            })
          });
          const data = await res.json();
          if (res.ok && data.success) {
            showModalAlert(`Cadastro realizado com sucesso! ID #${data.customer_id}.`, "success");
            setTimeout(() => {
              closeCustomerModal();
              appendMsg(`Olá! Concluí meu cadastro na loja. Meu nome é ${name} e meu contato é ${phone || email}.`, "user");
              sendMessageDirect(`Olá! Concluí meu cadastro na loja. Meu nome é ${name} e meu contato é ${phone || email}.`);
            }, 1200);
          } else {
            showModalAlert(data.detail || data.message || "Erro ao realizar cadastro.", "error");
          }
        } catch (e) {
          showModalAlert("Erro ao conectar com o servidor.", "error");
        }
      } else {
        const identifier = document.getElementById("updIdentifier").value.trim();
        const cep = document.getElementById("updCep").value.trim();
        const street = document.getElementById("updAddress").value.trim();
        const number = document.getElementById("updNumber").value.trim();
        const neighborhood = document.getElementById("updNeighborhood").value.trim();
        const city = document.getElementById("updCity").value.trim();
        const state = document.getElementById("updState").value.trim();

        if (!identifier) {
          showModalAlert("Informe seu telefone ou e-mail cadastrado.", "error");
          return;
        }
        if (!street) {
          showModalAlert("Informe o novo endereço/rua.", "error");
          return;
        }

        let fullAddress = street;
        if (number) fullAddress += `, ${number}`;
        if (neighborhood) fullAddress += ` - ${neighborhood}`;
        if (state) fullAddress += ` - ${state}`;

        try {
          const res = await fetch("/api/customer/address", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              customer_identifier: identifier,
              address: fullAddress,
              cep: cep || null,
              city: city || "Campo Grande",
            })
          });
          const data = await res.json();
          if (res.ok && data.success) {
            showModalAlert("Endereço atualizado com sucesso no cadastro!", "success");
            setTimeout(() => {
              closeCustomerModal();
              appendMsg(`Atualizei meu endereço de entrega para: ${fullAddress}.`, "user");
              sendMessageDirect(`Atualizei meu endereço de entrega para: ${fullAddress}.`);
            }, 1200);
          } else {
            showModalAlert(data.detail || data.message || "Cadastro não encontrado.", "error");
          }
        } catch (e) {
          showModalAlert("Erro ao conectar com o servidor.", "error");
        }
      }
    }

    async function sendMessageDirect(text) {
      try {
        const cartPayload = cart.map(i => ({
          product_id: i.product.product_id,
          name: i.product.name,
          price_brl: i.product.price_brl,
          effective_price_brl: i.product.effective_price_brl || i.product.price_brl,
          qty: i.qty
        }));
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: sessionId, cart: cartPayload })
        });
        const data = await res.json();
        if (data.action === "ADD_TO_CART" && data.product) {
          const p = data.product;
          const addQty = parseInt(data.qty, 10) || 1;
          const existing = cart.find(item => item.product.product_id === p.product_id);
          if (existing) {
            existing.qty = Math.min(p.stock_quantity || 99, existing.qty + addQty);
          } else {
            cart.push({ product: p, qty: addQty });
          }
          updateCartUI();
        } else if (data.action === "CLEAR_CART" || data.model_used === "direct_checkout_confirmed") {
          cart = [];
          updateCartUI();
        }
        appendMsg(data.reply, "bot", data);
      } catch (e) {
        appendMsg("Erro ao comunicar com o servidor.", "bot");
      }
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
                cart=payload.cart,
            )
            return ChatResponse(
                session_id=payload.session_id,
                reply=result["reply"],
                language=result["language"],
                confidence=result["confidence"],
                model_used=result["model_used"],
                tools_called=result.get("tools_called", []),
                products=result.get("products", []),
                active_agent=result.get("active_agent", "catalog"),
                action=result.get("action"),
                product=result.get("product"),
                qty=result.get("qty"),
                options=result.get("options", []),
                prefill=result.get("prefill"),
                logged_failed_intent=result.get("logged_failed_intent", False),
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
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
                active_agent="orchestrator",
                action=None,
                options=[],
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

    @app.get("/api/cep/{cep}", tags=["Customers & Addresses"])
    def lookup_cep_endpoint(cep: str) -> Dict[str, Any]:
        """Search Brazilian postal code (CEP) via ViaCEP API with offline cache fallback."""
        return CepService.lookup_cep(cep)

    @app.post("/api/customer/register", tags=["Customers & Addresses"])
    def register_customer_endpoint(payload: CustomerRegisterRequest) -> Dict[str, Any]:
        """Register a new customer and auto-enrich address using CEP lookup."""
        if not payload.name or not payload.name.strip():
            raise HTTPException(status_code=400, detail="Customer name cannot be empty.")
        if not payload.phone and not payload.email:
            raise HTTPException(status_code=400, detail="Must provide at least phone number or email.")
        return chat_service.agent.tool_executor.register_customer(
            name=payload.name,
            phone=payload.phone,
            email=payload.email,
            cep=payload.cep,
            address=payload.address,
            city=payload.city,
        )

    @app.post("/api/customer/address", tags=["Customers & Addresses"])
    def update_customer_address_endpoint(payload: CustomerAddressRequest) -> Dict[str, Any]:
        """Update delivery address and optional CEP for an existing customer."""
        if not payload.customer_identifier or not payload.customer_identifier.strip():
            raise HTTPException(status_code=400, detail="customer_identifier is required.")
        if not payload.address or not payload.address.strip():
            raise HTTPException(status_code=400, detail="Address cannot be empty.")
        result = chat_service.agent.tool_executor.update_customer_address(
            customer_identifier=payload.customer_identifier,
            address=payload.address,
            cep=payload.cep,
            city=payload.city,
        )
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message", "Customer not found."))
        return result

    @app.get("/api/pix-qrcode", tags=["Payments"])
    def get_pix_qrcode_endpoint(
        amount: str = "0.00",
        name: str = "Empório da Música",
        txid: str = "EMP123456",
    ) -> Response:
        """Generate and serve simulated SVG QR Code for PIX payments."""
        svg_content = PixService.generate_pix_svg(amount=amount, name=name, txid=txid)
        return Response(content=svg_content, media_type="image/svg+xml")

    return app


app = create_app()
