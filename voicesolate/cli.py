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
    parser.add_argument("-s", "--script", default=None, help="Script path (.txt, .json, .srt), URL, or episode ID")
    parser.add_argument("--provider", default=None, help="Optional script provider (e.g. 'startrek')")
    parser.add_argument("-o", "--output-dir", default="./output", help="Directory where character audio clips will be saved")
    parser.add_argument("-c", "--character", nargs="+", default=None, help="Specific character(s) to export (bypasses interactive TUI)")
    parser.add_argument("--wyoming-host", default="10.0.2.141", help="Wyoming STT server IP/hostname (default: 10.0.2.141)")
    parser.add_argument("--wyoming-port", type=int, default=10300, help="Wyoming STT server port (default: 10300)")
    parser.add_argument("--min-duration", type=float, default=5.0, help="Minimum clip duration in seconds (default: 5.0 to discard short utterances <= 5s, pass 0 to keep all)")
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
        border_style="cyan"
    ))

    # Optional connection check to Wyoming STT
    stt_client = WyomingSTTClient(host=args.wyoming_host, port=args.wyoming_port)
    try:
        console.print(f"[blue]Testing connection to Wyoming STT ({args.wyoming_host}:{args.wyoming_port})...[/blue]")
        stt_client.check_health()
        console.print("[green]✓ Connected to Wyoming STT (Whisper)[/green]\n")
    except Exception as e:
        console.print(f"[yellow]Notice: Wyoming STT not reachable ({e}). Using local faster-whisper.[/yellow]\n")
        stt_client = None

    # Get Media Duration
    media_duration = extractor.get_duration()
    console.print(f"[blue]Media duration:[/blue] {media_duration:.2f}s ({media_duration/60:.1f} minutes)")

    # Fetch & Parse Script
    parser = ScriptParser()
    script_source = args.script
    provider = args.provider

    # Generic Script Auto-detection
    if not script_source:
        # Check if user specified a provider or if Star Trek pattern is obvious
        if provider == "startrek" or "star trek" in episode_name.lower():
            m = re.search(r"s(\d{2})e(\d{2})", episode_name.lower())
            if m:
                script_source = f"s{m.group(1)}e{m.group(2)}"
                provider = "startrek"
                console.print(f"[cyan]Detected Star Trek episode script:[/cyan] {script_source}")
            elif "times arrow" in episode_name.lower():
                script_source = "s06e01" if "part 2" in episode_name.lower() else "s05e26"
                provider = "startrek"
                console.print(f"[cyan]Detected Star Trek episode script:[/cyan] {script_source}")

        if not script_source:
            # Universal fallback: Check for embedded subtitles in video file
            temp_srt = str(output_base_dir / "embedded_subs.srt")
            if extractor.extract_embedded_subtitles(temp_srt):
                script_source = temp_srt
                console.print("[cyan]Found and extracted embedded subtitles from video.[/cyan]")
            else:
                console.print("[yellow]No script specified. Please provide a script file, URL, or subtitle path with --script <file_or_url>[/yellow]")
                sys.exit(1)

    console.print(f"[cyan]Retrieving script from:[/cyan] {script_source}...")
    dialogues, is_cached = parser.fetch_or_load(script_source, provider=provider)
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

        # Filter clips by minimum duration if specified
        if args.min_duration > 0.0:
            initial_count = len(aligned_clips)
            aligned_clips = [c for c in aligned_clips if (c.end_sec - c.start_sec) >= args.min_duration]
            discarded = initial_count - len(aligned_clips)
            if discarded > 0:
                console.print(f"[yellow]Filtered out {discarded} clip(s) shorter than {args.min_duration:.1f}s.[/yellow]")

        export_task = progress.add_task("[magenta]Exporting & Enhancing Clips...", total=len(aligned_clips))

        for clip in aligned_clips:
            char_dir = output_base_dir / clip.character
            raw_dir = char_dir / "raw"
            enhanced_dir = char_dir / "enhanced"
            raw_dir.mkdir(parents=True, exist_ok=True)
            if not args.no_enhance:
                enhanced_dir.mkdir(parents=True, exist_ok=True)

            clip_path = raw_dir / f"{clip.timecode_str}.wav"
            enhanced_path = enhanced_dir / f"{clip.timecode_str}_enhanced.wav"

            # 1. Slice audio
            extractor.export_clip(clip.start_sec, clip.end_sec, str(clip_path))

            # 2. ML Cleanup & Super-Resolution Enhancement
            if not args.no_enhance:
                enhancer.clean_and_enhance_file(str(clip_path), str(enhanced_path))

            clip_entry = {
                "character": clip.character,
                "text": clip.text,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "confidence": clip.confidence,
                "file": str(clip_path)
            }
            if not args.no_enhance:
                clip_entry["enhanced_file"] = str(enhanced_path)

            manifest["clips"].append(clip_entry)

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
