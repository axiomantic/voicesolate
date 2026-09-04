# waveform.py - studio macro envelope generator
import os
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf

def extract_peaks_from_wav(wav_path: Path, num_points: int = 1200) -> Dict[str, Any]:
    if not wav_path.exists():
        return {'duration': 0.0, 'peaks': [0.0] * num_points, 'rms': [0.0] * num_points}

    try:
        info = sf.info(str(wav_path))
        total_frames = info.frames
        duration = info.duration
        samplerate = info.samplerate
        
        if total_frames <= 0 or duration <= 0:
            return {'duration': 0.0, 'peaks': [0.0] * num_points, 'rms': [0.0] * num_points}

        frames_per_point = max(1, total_frames // num_points)
        peaks = []
        rms_list = []

        with sf.SoundFile(str(wav_path)) as f:
            for _ in range(num_points):
                chunk = f.read(frames_per_point)
                if len(chunk) == 0:
                    peaks.append(0.0)
                    rms_list.append(0.0)
                    continue
                if chunk.ndim > 1:
                    chunk = chunk[:, 0]
                
                peak = float(np.max(np.abs(chunk)))
                rms = float(np.sqrt(np.mean(chunk**2)))
                peaks.append(min(1.0, round(peak, 4)))
                rms_list.append(min(1.0, round(rms * 1.8, 4)))

        while len(peaks) < num_points:
            peaks.append(0.0)
            rms_list.append(0.0)

        return {
            'duration': round(duration, 2),
            'samplerate': samplerate,
            'points': len(peaks),
            'peaks': peaks,
            'rms': rms_list
        }
    except Exception as e:
        return {'duration': 0.0, 'peaks': [0.0] * num_points, 'rms': [0.0] * num_points, 'error': str(e)}

def generate_macro_waveform_from_manifest(
    manifest_path: Path,
    num_points: int = 1200,
    estimated_duration: Optional[float] = None
) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {'duration': 0.0, 'points': num_points, 'peaks': [0.0] * num_points, 'clips': []}

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception:
        return {'duration': 0.0, 'points': num_points, 'peaks': [0.0] * num_points, 'clips': []}

    clips = manifest.get('clips', [])
    max_sec = 0.0
    for c in clips:
        end = c.get('end_sec', 0.0)
        if end > max_sec:
            max_sec = end

    total_duration = max(max_sec + 60.0, estimated_duration or 2700.0)
    
    episode_name = manifest.get('episode', '')
    cache_audio = Path('cache/audio') / f'{episode_name}.wav'
    raw_clips = list(manifest_path.parent.glob('**/raw/*.wav'))
    
    waveform_data = None
    if cache_audio.exists():
        waveform_data = extract_peaks_from_wav(cache_audio, num_points=num_points)
        total_duration = waveform_data['duration']
    elif raw_clips:
        peaks = [0.02] * num_points
        rms_list = [0.01] * num_points
        
        for c in clips:
            c_start = c.get('start_sec', 0.0)
            c_end = c.get('end_sec', 0.0)
            p_start = int((c_start / total_duration) * num_points)
            p_end = min(num_points - 1, int((c_end / total_duration) * num_points))
            
            conf = c.get('confidence', 85.0) / 100.0
            for idx in range(p_start, p_end + 1):
                t_ratio = (idx - p_start) / max(1, p_end - p_start)
                mod = 0.4 + 0.5 * math.sin(t_ratio * math.pi * 6.0) ** 2
                peaks[idx] = max(peaks[idx], round(min(1.0, mod * conf), 3))
                rms_list[idx] = max(rms_list[idx], round(min(1.0, peaks[idx] * 0.6), 3))

        waveform_data = {
            'duration': round(total_duration, 2),
            'samplerate': 16000,
            'points': num_points,
            'peaks': peaks,
            'rms': rms_list
        }
    else:
        waveform_data = {
            'duration': round(total_duration, 2),
            'samplerate': 16000,
            'points': num_points,
            'peaks': [0.05] * num_points,
            'rms': [0.02] * num_points
        }

    timeline_clips = []
    for idx, c in enumerate(clips):
        start = c.get('start_sec', 0.0)
        end = c.get('end_sec', 0.0)
        char = c.get('character', 'UNKNOWN')
        text = c.get('text', '')
        conf = round(c.get('confidence', 90.0), 1)
        raw_file = c.get('file', '')
        enh_file = c.get('enhanced_file', '')

        timeline_clips.append({
            'id': f'clip_{idx+1}',
            'character': char,
            'start_sec': round(start, 2),
            'end_sec': round(end, 2),
            'duration': round(end - start, 2),
            'start_ratio': round(start / total_duration, 4) if total_duration > 0 else 0,
            'end_ratio': round(end / total_duration, 4) if total_duration > 0 else 0,
            'confidence': conf,
            'text': text,
            'has_raw': bool(raw_file and Path(raw_file).exists()),
            'has_enhanced': bool(enh_file and Path(enh_file).exists()),
            'raw_file': raw_file,
            'enhanced_file': enh_file
        })

    waveform_data['clips'] = timeline_clips
    waveform_data['episode'] = episode_name
    return waveform_data
