import questionary
from rich.console import Console
from rich.table import Table
from typing import List
from .script_parser import CharacterStats

console = Console()

def display_character_table(characters: List[CharacterStats]):
    """Renders a Rich table of characters sorted by words spoken."""
    table = Table(title="🎭 Script Characters (Ranked by Total Words Spoken)")
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Character", style="bold green")
    table.add_column("Lines Spoken", justify="right", style="magenta")
    table.add_column("Total Words", justify="right", style="yellow")

    for rank, char in enumerate(characters, start=1):
        table.add_row(
            str(rank),
            char.name,
            f"{char.line_count:,}",
            f"{char.word_count:,}"
        )

    console.print()
    console.print(table)
    console.print()

def prompt_character_selection(characters: List[CharacterStats], default_all: bool = False) -> List[str]:
    """
    Shows an interactive TUI checkbox list for the user to select characters to export.
    Returns list of chosen character names.
    """
    if not characters:
        console.print("[red]No characters found in script![/red]")
        return []

    display_character_table(characters)

    choices = [
        questionary.Choice(
            title=f"{char.name:<25} ({char.word_count} words in {char.line_count} lines)",
            value=char.name,
            checked=default_all
        )
        for char in characters
    ]

    selected = questionary.checkbox(
        "Select characters to extract audio clips for (Space to toggle, Enter to confirm):",
        choices=choices,
        style=questionary.Style([
            ('checkbox-selected', 'fg:ansigreen bold'),
            ('selected', 'fg:ansicyan bold'),
            ('highlighted', 'fg:ansiyellow bold'),
            ('answer', 'fg:ansigreen bold'),
        ])
    ).ask()

    return selected or []
