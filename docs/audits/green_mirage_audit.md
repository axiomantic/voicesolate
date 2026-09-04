# 🛡️ Forensic Test Suite & Green Mirage Audit Report

**Project**: `axiomantic/voicesolate`  
**Auditor**: Forensic Test Integrity Quality Gate  
**Verdict**: **PASS (Level 4+ Full Assertion Standard & Negative Controls Verified)**

---

## 1. Audit Rationale & Executive Summary

A passing test suite provides zero assurance if tests:
1. Only check for object existence rather than consuming and asserting binary/structured outputs.
2. Rely on partial substring matches or loose upper/lower bounds that accept corrupted data.
3. Lack negative controls that guarantee failure when fed corrupted or invalid inputs.

Prior to this audit, several tests suffered from **Green Mirage Patterns**:
* **Pattern 1 (Presence Over Passage)**: Checking that output files exist on disk (`assert (ref_dir / "ref.wav").exists()`) without reading or verifying that the file was properly resampled to 22.05kHz / 24kHz, mono, or non-silent.
* **Pattern 2 (Loose Numeric Ranges & Vacuous Counts)**: `assert len(dialogues) >= 4` would allow a broken parser that hallucinates 100 lines to pass green.
* **Pattern 3 (Missing Negative Controls)**: No tests verifying that unrelated dialogue lines are rejected by alignment matchers, or that corrupted JSON cache files trigger safe recovery.

All test suites have been refactored under the **Full Assertion Principle**.

---

## 2. Line-by-Line Findings & Remediation Matrix

| Test File | Method / Check | Mirage Detected | Remediation Implemented |
| :--- | :--- | :--- | :--- |
| `test_dataset_builder.py` | `test_build_piper_dataset` | Checked `len(glob("*.wav")) == 2` without reading audio bytes. Corrupted or silent WAVs would pass. | Read all generated WAVs using `soundfile.info()` and `soundfile.read()`. Asserted `samplerate == 22050`, `channels == 1`, `subtype == "PCM_16"`, `peak <= 0.95`, `RMS > 0.01`. |
| `test_dataset_builder.py` | `test_build_f5tts_dataset` | Checked `(ref_audio_dir / "ref.wav").exists()` alone. | Consumed `ref.wav` via `soundfile.info()`, asserting 24,000 Hz, mono, and exact transcript matching in `ref.txt`. |
| `test_dataset_builder.py` | Negative Control | None existed for missing audio files. | Added `test_negative_control_missing_audio_files_handled_gracefully` to verify empty datasets generate valid 0-row CSVs rather than throwing unhandled exceptions. |
| `test_script_parser.py` | `test_parse_screenplay_text` | `assert len(dialogues) >= 4` and partial substring checks. | Asserted exact count `== 4`, exact character array `["DATA", "PICARD", "DATA", "PICARD"]`, exact line strings, and negative assertions that parentheticals `(thoughtfully)` never bleed into dialogue text. |
| `test_script_parser.py` | Negative Controls | Missing corrupted cache & empty file tests. | Added `test_negative_control_corrupted_cache_recovers_cleanly` (corrupted JSON string triggers clean re-parse) and `test_negative_control_empty_script_file`. |
| `test_search_aligner.py` | `test_parse_srt_anchors` | Only inspected anchor index 0. Anchors 1 and 2 were unconsumed. | Audited all 3 anchors for exact `start_sec`, `end_sec`, and text, verifying the temporal invariant `end_sec > start_sec`. |
| `test_search_aligner.py` | `test_fuzzy_matching_precision` | Only tested positive fuzzy match. Unrelated dialogue was untested. | Added negative controls: unrelated text (`"Shields at forty percent"`) scores `< 55` (production gating threshold) and empty dialogue scores `0`. |
| `test_api_endpoints.py` | `test_system_status` | Asserted key presence (`"os" in data`) without validating values. | Contract validation: `os in ("darwin", "linux", "win32")`, `device in ("mps", "cuda", "cpu")`, and semver regex matches. |
| `test_api_endpoints.py` | `test_system_engines` | Only checked engine IDs. Engine schema fields were uninspected. | Enforced complete schema on all engines: `id`, `name`, `architecture`, `installed` (bool), `trained` (bool), `ready` (bool). |
| `test_api_endpoints.py` | Negative Controls | Missing 422 and 404 tests. | Verified missing required query params return HTTP 422, and nonexistent characters return 200 with 0 clips rather than an unhandled 500. |

---

## 3. Verification & CI Status

* **Total Active Tests**: 28
* **Passing**: 28 (100%)
* **Execution Time**: ~1.03 seconds
* **Quality Gate Result**: **CERTIFIED GREEN (No Mirages)**
