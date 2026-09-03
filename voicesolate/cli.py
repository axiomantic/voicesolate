#!/usr/bin/env python3
import os
import sys
import re
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .wyoming_client import WyomingSTTClient
from .script_parser import ScriptParser
from .tui import prompt_character_selection, display_character_table
from .audio_extractor import AudioExtractor
from .search_aligner import SearchAligner
from .audio_enhancer import AudioEnhancer
from .cache_manager import CacheManager

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and enhance character voice audio from video/audio files using Wyoming STT and ML models."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input video or audio file (e.g. episode.mkv)")
    parser.add_argument("-s", "--script", default=None, help="Script path, URL, or Star Trek episode ID (e.g. 's05e26' or 'Time\\'s Arrow')")
    parser.add_argument("-o", "--output-dir", default="./output", help="Directory where character audio clips will be saved")
    parser.add_argument("-c", "--character", nargs="+", default=None, help="Specific character(s) to export (bypasses interactive TUI)")
    parser.add_argument("--wyoming-host", default="10.0.2.141", help="Wyoming STT server IP/hostname (default: 10.0.2.141)")
    parser.add_argument("--wyoming-port", type=int, default=10300, help="Wyoming STT server port (default: 10300)")
    parser.add_argument("--no-enhance", action="store_true", help="Skip ML vocal isolation and super-resolution enhancement")
    parser.add_argument("--all-characters", action="store_true", help="Select all characters found in script")
    return parser.parse_args()

def main():
    args = parse_args()
    input_str = args.input.strip()
    
    # Initialize Audio Extractor (handles local or remote SFTP/SSH)
    extractor = AudioExtractor(input_str)
    
    # Compute episode name
    if extractor.is_remote:
        filename = Path(extractor.remote_file_path).stem
        display_input = f"{extractor.remote_user}@{extractor.remote_host}:{extractor.remote_file_path}"
    else:
        filename = extractor.local_path.stem
        display_input = str(extractor.local_path)

    # Sanitize episode name for folder creation
    episode_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", filename)[:60]
    output_base_dir = Path(args.output_dir).resolve() / episode_name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        f"[bold cyan]🎙️ Voicesolate — Character Dialogue Extraction & Studio Isolation[/bold cyan]\n"
        f"[yellow]Input Source:[/yellow] {display_input}\n"
        f"[yellow]Mode:[/yellow] {'Remote Zero-Download Chunk Seeking (SSH/SFTP)' if extractor.is_remote else 'Local Audio Seeking'}\n"
        f"[yellow]Wyoming Server:[/yellow] {args.wyoming_host}:{args.wyoming_port}\n"
        f"[yellow]Output Target:[/yellow] {output_base_dir}",
        title="Configuration"
    ))

    # Test Wyoming STT connectivity
    console.print(f"[cyan]Testing connection to Wyoming STT ({args.wyoming_host}:{args.wyoming_port})...[/cyan]")
    stt_client = WyomingSTTClient(host=args.wyoming_host, port=args.wyoming_port)
    try:
        health = stt_client.check_health()
        asr_info = health.get("asr", [{}])[0].get("name", "Whisper")
        console.print(f"[green]✓ Connected to Wyoming STT ({asr_info})[/green]\n")
    except Exception as e:
        console.print(f"[bold red]✗ Failed to connect to Wyoming server:[/bold red] {e}")
        console.print("[yellow]Please check server address and port.[/yellow]")
        sys.exit(1)

    # Get Media Duration
    media_duration = extractor.get_duration()
    console.print(f"[blue]Media duration:[/blue] {media_duration:.2f}s ({media_duration/60:.1f} minutes)")

    # Fetch & Parse Script
    parser = ScriptParser()
    script_source = args.script

    # Auto-detect script if not provided
    if not script_source:
        # Check if episode name matches pattern (e.g. s06e01 or s05e26)
        m = re.search(r"s(\d{2})e(\d{2})", episode_name.lower())
        if m:
            script_source = f"s{m.group(1)}e{m.group(2)}"
            console.print(f"[cyan]Auto-detected Star Trek episode script:[/cyan] {script_source}")
        elif "times arrow part 2" in episode_name.lower() or "part 2" in episode_name.lower():
            script_source = "s06e01"
            console.print(f"[cyan]Auto-detected Star Trek episode script:[/cyan] {script_source}")
        elif "times arrow" in episode_name.lower():
            script_source = "s05e26"
            console.print(f"[cyan]Auto-detected Star Trek episode script:[/cyan] {script_source}")
        else:
            # Check for embedded subtitles
            temp_srt = str(output_base_dir / "embedded_subs.srt")
            if extractor.extract_embedded_subtitles(temp_srt):
                script_source = temp_srt
                console.print("[cyan]Found and extracted embedded subtitles from video.[/cyan]")
            else:
                console.print("[yellow]No script provided. Please specify --script <file_or_id> or URL.[/yellow]")
                sys.exit(1)

    console.print(f"[cyan]Retrieving script from:[/cyan] {script_source}...")
    dialogues, is_cached = parser.fetch_or_load(script_source)
    if is_cached:
        console.print("[green]✓ Loaded from local script cache (0 network latency)[/green]")
    else:
        console.print("[green]✓ Script fetched and cached locally[/green]")

    sorted_characters = parser.get_characters_sorted()
    if not sorted_characters:
        console.print("[bold red]Error: No dialogue or characters could be extracted from the script.[/bold red]")
        sys.exit(1)

    # Character Selection via TUI or Args
    if args.character:
        selected_characters = [c.upper() for c in args.character]
        display_character_table([c for c in sorted_characters if c.name in selected_characters])
    elif args.all_characters:
        selected_characters = [c.name for c in sorted_characters]
        display_character_table(sorted_characters)
    else:
        selected_characters = prompt_character_selection(sorted_characters)

    if not selected_characters:
        console.print("[yellow]No characters selected. Exiting.[/yellow]")
        return

    # Check or extract subtitle anchors
    subtitles_cache = Path("cache/subtitles") / f"{episode_name}.srt"
    subtitles_cache.parent.mkdir(parents=True, exist_ok=True)
    if not subtitles_cache.exists():
        console.print("[cyan]Extracting subtitle anchors for instant timecode alignment...[/cyan]")
        if extractor.extract_embedded_subtitles(str(subtitles_cache)):
            console.print("[green]✓ Subtitle anchors extracted successfully[/green]")
        else:
            subtitles_cache = None

    # Hierarchical STT Alignment & Slicing
    aligner = SearchAligner(extractor, stt_client)
    enhancer = AudioEnhancer()

    manifest = {
        "episode": episode_name,
        "selected_characters": selected_characters,
        "clips": []
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        aligned_clips = aligner.align_character_lines(
            all_script_lines=parser.dialogues,
            target_characters=selected_characters,
            subtitles_path=str(subtitles_cache) if subtitles_cache and subtitles_cache.exists() else None,
            script_id=str(script_source),
            progress=progress
        )

        export_task = progress.add_task("[magenta]Exporting & Enhancing Clips...", total=len(aligned_clips))

        for clip in aligned_clips:
            char_dir = output_base_dir / clip.character
            char_dir.mkdir(parents=True, exist_ok=True)
            clip_path = char_dir / f"{clip.timecode_str}.wav"

            # 1. Slice audio
            extractor.export_clip(clip.start_sec, clip.end_sec, str(clip_path))

            # 2. ML Cleanup & Super-Resolution Enhancement
            if not args.no_enhance:
                enhanced_path = char_dir / f"{clip.timecode_str}_enhanced.wav"
                enhancer.clean_and_enhance_file(str(clip_path), str(enhanced_path))

            manifest["clips"].append({
                "character": clip.character,
                "text": clip.text,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "confidence": clip.confidence,
                "file": str(clip_path)
            })

            progress.advance(export_task, 1)

    # Save manifest.json
    manifest_file = output_base_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"\n[bold green]✓ Voice extraction complete![/bold green]")
    console.print(f"[cyan]Total clips extracted:[/cyan] {len(manifest['clips'])}")
    console.print(f"[cyan]Output folder:[/cyan] {output_base_dir}")
    console.print(f"[cyan]Manifest file:[/cyan] {manifest_file}")

if __name__ == "__main__":
    main()
