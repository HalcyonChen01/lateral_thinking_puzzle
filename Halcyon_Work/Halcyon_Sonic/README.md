# Halycon_Music Project

This is the README for the Halycon_Music project.

Audio-driven visuals (microphone & audio-file)

Overview
--------
This is a single-file Python project (`Halycon_Music.py`) that maps audio input (live microphone or pre-analyzed audio files) to several visual styles: RINGS, SONIC_SPHERE and PAINT_STRIPES. The project demonstrates real-time audio analysis, band-energy driven visuals, and several rendering/compositing techniques implemented with pygame.

Key features
------------
- Live microphone input (uses `sounddevice`; if not available the code falls back to a built-in simulator).
- Audio-file analysis (e.g. MP3) using `librosa` to precompute low/mid/high band energies; optional playback via `pygame.mixer`.
- Three visual modes:
  - RINGS: concentric audio-driven expanding rings with highlight edges.
  - SONIC_SPHERE: a spectrum-driven sphere centered on the screen.
  - PAINT_STRIPES: oil-paint style horizontal stripes with a short history trail, blur and drip effects.
- Basic UI controls: mode selection, file upload, play/pause/stop and explicit "Use MIC" to restore live input.

Dependencies & environment
-------------------------
Create and use a virtual environment (venv or conda recommended). Development and testing happened primarily on Windows (PowerShell).

Required Python packages:
- pygame
- numpy
- librosa

Optional (for live microphone input):
- sounddevice  (if absent, the app falls back to a simulator for visuals)

Example install (recommended inside a virtual environment)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pygame numpy librosa
# For microphone support:
pip install sounddevice
```

How to run
----------
Run the script from the directory that contains `Halycon_Music.py` (PowerShell example):
```powershell
# Activate virtual environment (if using .venv)
.\.venv\Scripts\Activate.ps1
python "Halycon_Music.py"
```
A pygame window will open and visuals will start. By default the program attempts to start microphone analysis; if `sounddevice` is not available, a simulator is used so visuals can still be tested.

File input & playback
---------------------
- Use the Upload button in the UI, or press `F`, to pick an audio file (requires tkinter for the file dialog to appear).
- Loaded files are analyzed with `librosa` (STFT) to produce frame-wise low/mid/high energy arrays; visuals use those precomputed energies.
- If `pygame.mixer` can play the file, the program will attempt playback. Playback and pause affect visuals: when in FILE mode and playback is paused, visuals are frozen (no new spawns, time frozen, history preserved) until playback resumes or the user explicitly restores the microphone.

Input mode switching
--------------------
- The application supports two input modes: `MIC` (live microphone) and `FILE` (loaded audio file).
- When switching to `FILE`, the microphone analyzer is paused/stopped; the microphone is not auto-restored when the file stops — use the UI "Use MIC" button or the `U` key to switch back to live input.

Visual modes & controls
----------------------
- Switch visual modes via the UI buttons or the numeric keys 1/2/3.
- The PAINT_STRIPES mode keeps a short history (approx. 3 seconds) and has tunable parameters defined as constants at the top of the script (stroke thickness, stripe width, fill behavior, etc.).

Code structure & core points
--------------------------
- Single-file implementation: `Halycon_Music.py`. Major parts:
  - Audio analysis: `analyze_audio()` (librosa) and `MicrophoneAnalyzer` (sounddevice-based rFFT, noise learning and spectral subtraction).
  - Visual rendering: `draw_single_ring()`, `draw_sonic_sphere()`, `draw_painted_stripes()`.
  - Main loop `main()`: event handling, state management (`running`, `file_paused`, `mic_paused`, `INPUT_MODE`) and per-frame updates.

Concepts (brief)
----------------
- Band separation: splitting audio into low/mid/high bands and using those band energies to drive visual parameters.
- Time-frequency analysis: STFT (short-time Fourier transform) to obtain frequency energy over time (implemented with librosa).
- Live vs precomputed analysis: microphone mode computes energies in near-real time; file mode precomputes the whole feature matrix for stable visual playback.
- Visual-freeze strategy: when a file playback is paused the program freezes animation time and stops spawning new visual elements so the image holds steady.

FAQ
---
- Q: Why doesn't the microphone work?
  A: Install `sounddevice` and ensure Python has permission to access the recording device. If unavailable the program uses a simulator so visuals still function.

- Q: Can I use MP4 or other non-standard encodings?
  A: `librosa` can read most audio containers (it relies on underlying codecs like ffmpeg); playback is handled by `pygame.mixer` so some formats may not play but can still be used for visual analysis.

References
----------
- librosa: McFee, B., Raffel, C., et al., "librosa: Audio and Music Signal Analysis in Python".
- pygame: pygame.org — cross-platform game/multimedia library.
