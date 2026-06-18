# Scenic Montage Generator

Generate ~30s dramatic montage videos from YouTube clips, beat-synced to any song.

## Requirements

- Python 3.14+ (via [uv](https://docs.astral.sh/uv/))
- [ffmpeg](https://ffmpeg.org/) — `brew install ffmpeg`
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — `brew install yt-dlp`

## Setup

```bash
uv sync
```

## Usage

### Single montage

```bash
uv run montage.py --topic "boston trains" --song "for the heavens by axjunior"
```

Or with the natural language query format:

```bash
uv run montage.py "boston trains, using song for the heavens by axjunior"
```

### Batch processing

Generate a CSV of 10 topic/song ideas:

```bash
uv run montage.py --init-csv
```

Then process all pending entries:

```bash
uv run montage.py --csv prompts.csv
```

Batch processing updates `prompts.csv` with status as each montage completes.

## Output structure

```
song__topic/
├── clips/                        # downloaded YouTube clips
│   ├── clip_01.mp4
│   └── ...
├── song_30s.wav                  # most energetic 30s of the song
└── song__topic.mp4               # final montage

songs/                            # shared song cache (reused across projects)
└── song_name.wav
```

## How it works

1. Downloads the song from YouTube and extracts audio
2. Scans the song for the most energetic 30-second segment (loudest RMS energy)
3. Detects beats with `librosa` and picks 4 evenly-spaced transition points
4. Downloads 5 video clips from YouTube matching the topic
5. Builds an ffmpeg complex filter graph: trims clips, crossfades at beat boundaries,
   scales to 1080p, fades audio/video in/out
6. Encodes to H.264 MP4 with AAC audio
