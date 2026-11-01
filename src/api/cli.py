"""Interactive Rich CLI interface for testing and conversing with the Empório da Música Agent."""

from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.services.chat_service import ChatService


console = Console()


def render_welcome_banner(session_id: str) -> None:
    """Display welcome banner and available slash commands."""
    banner_text = (
        "**🎸 Empório da Música — Assistente Virtual Inteligente**\n\n"
        "*Sua música começa aqui! (Campo Grande/MS)*\n\n"
        "Comandos rápidos:\n"
        "- `/reset`   : Limpar histórico da sessão atual\n"
        "- `/history` : Exibir histórico de mensagens da sessão\n"
        "- `/failed`  : Listar intenções ambíguas/baixa confiança registradas (`failed_intents`)\n"
        "- `/exit`    : Encerrar o atendimento"
    )
    console.print(
        Panel(
            Markdown(banner_text),
            title=f"[bold cyan]Sessão: {session_id}[/bold cyan]",
            border_style="cyan",
        )
    )


def run_cli(session_id: str = "cli_user_01") -> None:
    """Start the interactive command-line chat loop."""
    chat_service = ChatService()
    render_welcome_banner(session_id)

    while True:
        try:
            user_input = console.input("\n[bold green]Você > [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Encerrando atendimento. Até logo! 🎶[/yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "sair"):
            console.print("[yellow]Obrigado por visitar a Empório da Música! Até a próxima! 🎶[/yellow]")
            break

        if cmd == "/reset":
            chat_service.clear_session(session_id)
            console.print("[bold yellow]✔ Histórico da conversa reiniciado com sucesso.[/bold yellow]")
            continue

        if cmd == "/history":
            history = chat_service.get_history(session_id)
            if not history:
                console.print("[dim]Nenhuma mensagem registrada nesta sessão.[/dim]")
                continue
            table = Table(title=f"Histórico da Sessão ({session_id})")
            table.add_column("Turno", style="cyan", width=8)
            table.add_column("Papel", style="magenta", width=12)
            table.add_column("Mensagem", style="white")
            for idx, item in enumerate(history, start=1):
                table.add_row(str(idx), item["role"], item["content"][:120])
            console.print(table)
            continue

        if cmd == "/failed":
            failures = chat_service.agent.failed_intent_repo.get_all_failures(limit=10)
            if not failures:
                console.print("[dim]Nenhum evento de baixa confiança registrado até o momento.[/dim]")
                continue
            table = Table(title="Eventos de Baixa Confiança / Ambiguidade (FailedIntentRepository)")
            table.add_column("ID", style="cyan")
            table.add_column("Idioma", style="green")
            table.add_column("Motivo", style="yellow")
            table.add_column("Confiança", style="red")
            table.add_column("Mensagem", style="white")
            for f in failures:
                table.add_row(
                    str(f["id"]),
                    f["detected_language"],
                    f["reason"],
                    f"{f['confidence']:.2f}",
                    f["user_message"],
                )
            console.print(table)
            continue

        with console.status("[bold cyan]Consultando sistema da Empório da Música...[/bold cyan]"):
            result = chat_service.process_message(user_input, session_id=session_id)

        tools_info = ", ".join(t["tool"] for t in result.get("tools_called", [])) or "direct"
        subtitle = (
            f"[dim]Model: {result.get('model_used')} | "
            f"Lang: {result.get('language')} | "
            f"Tools: {tools_info}[/dim]"
        )

        console.print(
            Panel(
                Markdown(result["reply"]),
                title="[bold magenta]🎵 Empório da Música[/bold magenta]",
                subtitle=subtitle,
                border_style="magenta",
            )
        )
