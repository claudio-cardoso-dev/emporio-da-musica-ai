"""Conversational AI Agent with Groq Tool Calling, Model Fallbacks, and Ambiguity/Low-Confidence Guardrails."""

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
from groq import Groq

from src.agents.prompts import (
    LOW_CONFIDENCE_TEMPLATES,
    SYSTEM_PROMPT,
    detect_user_language,
)
from src.agents.tools import TOOL_SCHEMAS, ToolExecutor
from src.config.settings import settings
from src.repositories.failed_intent_repository import FailedIntentRepository
from src.repositories.product_repository import normalize_text


def clean_thinking_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks emitted by reasoning models like Qwen3."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


class AssistantAgent:
    """
    Orchestrates NLU analysis, ambiguity/low-confidence detection, multi-step Tool Calling,
    and LLM provider fallbacks (primary: qwen/qwen3-32b -> fallback 1: llama-3.3-70b-versatile -> fallback 2: llama-3.1-8b-instant).
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        pdf_path: Optional[Path] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.pdf_path = pdf_path
        self.api_key = api_key or settings.groq_api_key
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.tool_executor = ToolExecutor(db_path=db_path, pdf_path=pdf_path)
        self.failed_intent_repo = FailedIntentRepository(db_path=db_path)

        # Ordered model chain as requested: Qwen3 primary, Llama-3.3-70b and Llama-3.1-8b fallbacks
        self.model_chain = [
            settings.primary_model,
            settings.fallback_model_1,
            settings.fallback_model_2,
        ]

    def evaluate_clarity_and_confidence(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether the user message is intelligible, ambiguous, or clear.
        Low-confidence or gibberish inputs are intercepted at low temperature and logged
        to `failed_intents` to continuously improve NLU accuracy.
        """
        lang = detect_user_language(user_message)
        stripped = (user_message or "").strip()

        # 1. Empty or ultra-short non-word noise
        alnum_chars = re.sub(r"[^a-zA-Z0-9]", "", stripped)
        if len(alnum_chars) < 2:
            return {
                "is_clear": False,
                "reason": "too_short_or_empty",
                "confidence": 0.10,
                "language": lang,
                "strategy": "ask_rephrase_with_options",
            }

        # 2. Gibberish / keyboard smash heuristic (no vowels or random symbol clusters)
        norm = normalize_text(stripped)
        words = re.findall(r"[a-z]+", norm)
        vowels = set("aeiou")
        if words:
            vowel_words = [w for w in words if any(ch in vowels for ch in w)]
            if len(vowel_words) == 0 or (
                len(words) == 1 and len(words[0]) > 6 and len(set(words[0]).intersection(vowels)) <= 1
            ):
                return {
                    "is_clear": False,
                    "reason": "gibberish_or_unintelligible",
                    "confidence": 0.15,
                    "language": lang,
                    "strategy": "ask_rephrase_with_options",
                }
        elif not stripped.isdigit():
            return {
                "is_clear": False,
                "reason": "no_recognizable_words",
                "confidence": 0.15,
                "language": lang,
                "strategy": "ask_rephrase_with_options",
            }

        # 3. Context-free vague utterances when there is no prior conversation history
        vague_phrases = {
            "isso", "aquilo", "aquele", "negocio", "coisa", "quero", "preciso",
            "ajuda", "duvida", "nao sei", "hm", "hmm", "talvez", "qualquer um",
            "that thing", "something", "whatever", "idk", "eso", "aquello",
        }
        if not chat_history and norm in vague_phrases:
            return {
                "is_clear": False,
                "reason": "vague_without_context",
                "confidence": 0.35,
                "language": lang,
                "strategy": "disambiguation_menu",
            }

        return {
            "is_clear": True,
            "reason": "clear",
            "confidence": 0.92,
            "language": lang,
            "strategy": "proceed_to_agent",
        }

    def _call_llm_with_fallbacks(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
    ) -> tuple[Any, str]:
        """
        Invoke Groq Chat Completions trying `qwen/qwen3-32b` first and falling back
        to `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on rate limit or error.
        """
        if not self.client:
            raise RuntimeError("Groq API client is not initialized (missing GROQ_API_KEY).")

        temp = temperature if temperature is not None else settings.llm_temperature
        last_error: Optional[Exception] = None

        for model_name in self.model_chain:
            for attempt in range(2):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": 1100,
                    }
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = "auto"

                    response = self.client.chat.completions.create(**kwargs)
                    return response, model_name
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()
                    if ("429" in err_str or "413" in err_str or "rate_limit" in err_str) and attempt == 0:
                        time.sleep(2.5)
                        continue
                    break

        raise RuntimeError(f"All LLM models in fallback chain failed. Last error: {last_error}")

    def run(
        self,
        user_message: str,
        session_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message end-to-end:
        1. Check clarity & confidence; log to `failed_intents` if ambiguous/low-confidence.
        2. Build conversation context with language instruction.
        3. Execute multi-step tool calling loop (up to 4 iterations).
        4. Return structured response with metadata (model used, tools called, language).
        """
        history = chat_history or []
        clarity = self.evaluate_clarity_and_confidence(user_message, history)
        detected_lang = clarity["language"]

        if not clarity["is_clear"] or clarity["confidence"] < settings.confidence_threshold:
            self.failed_intent_repo.log_failure(
                session_id=session_id,
                user_message=user_message,
                detected_language=detected_lang,
                reason=clarity["reason"],
                confidence=clarity["confidence"],
                resolution_strategy=clarity["strategy"],
            )
            fallback_reply = LOW_CONFIDENCE_TEMPLATES.get(
                detected_lang, LOW_CONFIDENCE_TEMPLATES["pt"]
            )
            return {
                "reply": fallback_reply,
                "language": detected_lang,
                "confidence": clarity["confidence"],
                "model_used": "guardrail_low_confidence",
                "tools_called": [],
                "products": [],
                "logged_failed_intent": True,
            }

        lang_labels = {
            "pt": "Portuguese (Português Brasileiro)",
            "en": "English",
            "es": "Spanish (Español)",
        }
        lang_instruction = (
            f"\n\n[ACTIVE TURN DIRECTIVE]: The user's message is in {lang_labels.get(detected_lang, 'the same language as the user')}. "
            f"You MUST write your entire final response in that exact language. "
            f"If the request is ambiguous (e.g., multiple matching products, multiple customers with the same name, or unspecified exchange reason), "
            f"logically present the options to help the user disambiguate."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT + lang_instruction}
        ]

        # Append recent conversation history (up to last 10 messages)
        for msg in history[-10:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        tools_called_log: List[Dict[str, Any]] = []
        collected_products: Dict[int, Dict[str, Any]] = {}
        model_used = self.model_chain[0]

        for _ in range(4):
            response, model_used = self._call_llm_with_fallbacks(
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=settings.llm_temperature,
            )
            choice_msg = response.choices[0].message
            tool_calls = getattr(choice_msg, "tool_calls", None)

            if not tool_calls:
                final_text = clean_thinking_tags(choice_msg.content or "")
                return {
                    "reply": final_text,
                    "language": detected_lang,
                    "confidence": clarity["confidence"],
                    "model_used": model_used,
                    "tools_called": tools_called_log,
                    "products": list(collected_products.values()),
                    "logged_failed_intent": False,
                }

            # Append assistant message containing tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": choice_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    fn_args = {}

                tool_output = self.tool_executor.execute(fn_name, fn_args)
                tools_called_log.append(
                    {
                        "tool": fn_name,
                        "arguments": fn_args,
                    }
                )

                # Collect structured product cards for UI Carousel rendering
                if isinstance(tool_output, dict):
                    if "products" in tool_output and isinstance(tool_output["products"], list):
                        for prod in tool_output["products"]:
                            if isinstance(prod, dict) and "product_id" in prod:
                                collected_products[int(prod["product_id"])] = prod
                    if "product" in tool_output and isinstance(tool_output["product"], dict):
                        prod = tool_output["product"]
                        if "product_id" in prod:
                            collected_products[int(prod["product_id"])] = prod
                    if "recommended_alternatives_in_stock" in tool_output and isinstance(
                        tool_output["recommended_alternatives_in_stock"], list
                    ):
                        for alt in tool_output["recommended_alternatives_in_stock"]:
                            if isinstance(alt, dict) and "product_id" in alt:
                                full_alt = self.tool_executor.product_repo.get_by_id(
                                    int(alt["product_id"])
                                )
                                if full_alt:
                                    collected_products[int(full_alt["product_id"])] = full_alt

                # Check if tool returned an ambiguity flag (e.g., multiple customers with same name)
                if isinstance(tool_output, dict) and tool_output.get("ambiguous_customers"):
                    self.failed_intent_repo.log_failure(
                        session_id=session_id,
                        user_message=user_message,
                        detected_language=detected_lang,
                        reason="ambiguous_customer_name",
                        confidence=0.55,
                        resolution_strategy="present_customer_disambiguation_options",
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    }
                )

        # Final synthesis call without tools if loop exhausted
        response, model_used = self._call_llm_with_fallbacks(
            messages=messages,
            tools=None,
            temperature=settings.llm_temperature,
        )
        final_text = clean_thinking_tags(response.choices[0].message.content or "")
        return {
            "reply": final_text,
            "language": detected_lang,
            "confidence": clarity["confidence"],
            "model_used": model_used,
            "tools_called": tools_called_log,
            "products": list(collected_products.values()),
            "logged_failed_intent": False,
        }

