# Karaoke Creator

Create UltraStar Deluxe karaoke songs from any audio or video file.

## What it does

1. **Audio extraction** — extracts audio from video files using ffmpeg
2. **Vocal separation** — isolates vocals from instruments using Demucs (Meta)
3. **Lyrics alignment** — ForceAlign (Wav2Vec2) for precise word boundaries when lyrics are provided; Whisper fallback
4. **Pitch detection** — analyzes vocal pitch using CREPE neural network
5. **BPM detection** — detects song tempo using librosa
6. **UltraStar generation** — combines everything into a valid UltraStar Deluxe TXT file

## Prerequisites

- Python 3.11+
- Node.js 18+
- ffmpeg installed and available in PATH

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

For **precise word-level alignment** (recommended when pasting lyrics), ForceAlign is used by default. If pip fails with SSL errors, try:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org forcealign
# or: pip install git+https://github.com/lukerbs/forcealign.git
```

### Frontend

```bash
cd frontend
npm install
```

## Running

### Backend (port 8000)

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (port 5173)

```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

## Output format

The app generates a ZIP containing:
- `Artist - Title.txt` — UltraStar Deluxe song file
- `Artist - Title.mp3` — full audio mix
- `Artist - Title [VOC].mp3` — isolated vocals
- `Artist - Title [INSTR].mp3` — instrumental (karaoke) track

Place the extracted folder into your UltraStar Deluxe songs directory.

## Tech stack

- **Backend**: Python, FastAPI, Demucs, faster-whisper, CREPE, librosa
- **Frontend**: React, TypeScript, Vite, TailwindCSS
