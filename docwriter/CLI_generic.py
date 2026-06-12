import time
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()

# Claude-inspired color palette
ACCENT_COLOR = "#d97757"  # Clay / Terracotta
DIM_COLOR = "#9ca3af"     # Muted Gray
USER_COLOR = "#3b82f6"    # Subtle Blue for User


class ClaudeCLI:
    def __init__(self, app_name="LLMlight CLI"):
        self.app_name = app_name
        self.model = "google/gemma-26B-a4B"
        self.path = "C:/LLMlight"

    def welcome(self):
        console.print()

        # 1. Centered Header & ASCII Art
        ascii_logo = r"""
  .-.
 |o o|
 | = |
/|___|\
"""
        header_text = Text()
        header_text.append(f" {self.app_name} \n", style=f"bold {ACCENT_COLOR} reverse")
        header_text.append(ascii_logo, style=ACCENT_COLOR)
        header_text.append(f"\nmodel: ", style=f"dim {DIM_COLOR}")
        header_text.append(f"{self.model}\n", style="italic")
        header_text.append(f"path:  ", style=f"dim {DIM_COLOR}")
        header_text.append(f"{self.path}\n", style="dim")

        centered_header = Align.center(header_text, vertical="middle")

        # 2. Side-by-Side Content Grid (Fixed parsing using Text.from_markup)
        grid = Table.grid(expand=True, padding=(0, 4))
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        tips_text = Text.from_markup(f"""[bold {ACCENT_COLOR}]Getting started[/bold {ACCENT_COLOR}]

Type commands like:
  • [cyan]/init[/cyan]
  • [cyan]/help[/cyan]
  • generate documentation""")

        updates_text = Text.from_markup(f"""[bold {ACCENT_COLOR}]What’s new[/bold {ACCENT_COLOR}]

• Agent system improvements
• Documentation generation
• Local model support

[dim]/release-notes for details[/dim]""")

        grid.add_row(tips_text, updates_text)

        # 3. Layout layout container
        layout_group = Group(
            centered_header,
            "─" * 64,  # Clean internal divider line
            "",
            grid
        )

        # Main welcome card
        console.print(
            Align.center(
                Panel(
                    layout_group,
                    box=box.ROUNDED,
                    border_style=ACCENT_COLOR,
                    padding=(2, 4),
                    width=76
                )
            )
        )
        console.print()

    def user_message(self, text):
        console.print(f"\n[bold {USER_COLOR}]👤 You[/bold {USER_COLOR}] › {text}\n")

    def assistant_message(self, text):
        # Clean, distraction-free markdown block matching Claude's aesthetic
        console.print(f"[bold {ACCENT_COLOR}]🤖 Assistant[/bold {ACCENT_COLOR}]")
        console.print("═" * 40, style=ACCENT_COLOR)
        console.print(Markdown(text))
        console.print("═" * 40, style="dim")
        console.print()

    def status(self, text):
        with Live(
            Spinner("dots", text=f"[dim]{text}[/dim]", style=ACCENT_COLOR),
            refresh_per_second=12,
            transient=True
        ):
            time.sleep(1.2)

    def tool_call(self, tool_name):
        console.print(
            f"  [dim]🛠️ Running tool:[/dim] [italic cyan]{tool_name}[/italic cyan]..."
        )

    def error(self, text):
        console.print(
            f"[bold red]✕ Error:[/bold red] {text}"
        )

    def run(self):
        self.welcome()

        while True:
            try:
                prompt = Prompt.ask(f"[bold {ACCENT_COLOR}]❯[/bold {ACCENT_COLOR}]").strip()
                
                if not prompt:
                    continue

                if prompt.lower() in {"exit", "quit"}:
                    console.print(f"[dim]Goodbye! Thanks for using {self.app_name}.[/dim]")
                    break

                if prompt == "/help":
                    self.assistant_message(
                        """
### Available Commands
* `/help`          - Show this utility manual
* `/clear`         - Clear the current terminal window
* `quit` / `exit`  - Safely terminate the session
                        """
                    )
                    continue

                if prompt == "/clear":
                    console.clear()
                    continue

                self.user_message(prompt)
                self.status("Thinking...")
                
                reply = f"Successfully captured your request: **'{prompt}'**. Let me know how you want to expand this logic framework."
                self.assistant_message(reply)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Session terminated gracefully.[/dim]")
                break


if __name__ == "__main__":
    ClaudeCLI().run()