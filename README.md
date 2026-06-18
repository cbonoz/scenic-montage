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

Each run creates a `{topic}__{song}/` directory at the project root. Songs are cached in a shared `songs/` folder for reuse across projects.

```
boston_trains__for_the_heavens_by_axjunior/
├── clips/                                    # downloaded YouTube videos
│   ├── clip_01.mp4
│   └── ...
├── song_30s.wav                              # most energetic 30s of the song
└── boston_trains__for_the_heavens_by_axjunior.mp4   # final montage

songs/                                        # shared song cache (reused)
└── for_the_heavens_by_axjunior.wav
```

## How it works

1. **Song**: Downloads from YouTube via `yt-dlp`, cached in `songs/` as WAV
2. **Energy analysis**: Uses `librosa.feature.rms` to find the loudest 30-second segment
3. **Beat detection**: `librosa.beat.beat_track` detects ~137 BPM tempo, picks 4 evenly-spaced transition points
4. **Clip selection** (two-tier search):
   - Quick `--flat-playlist` scan of 50+ YouTube results across 3 queries (`cinematic`, `drone`, plain topic)
   - Filters by duration (20–300s), view count, and title keywords
   - Deep-probes top 10 candidates with `--dump-json` to get exact resolution and aspect ratio
   - Ranks by: 1080p > 720p > 480p, 16:9 landscape, penalizes >2K (huge files) and long duration
   - Downloads top 8 with `--max-filesize 200M` guard
5. **Filter graph**: ffmpeg complex filter trims clips, crossfades at beat boundaries, scales to 1080p, fades audio/video in/out
6. **Encoding**: libx264 (or h264_videotoolbox on macOS fallback) with AAC audio

## Tips for good results

- **Topics that work well**: cities, landscapes, travel, nature, architecture, "cinematic" stock footage style
- **Avoid topics**: specific people/events (copyright), fast action sports (hard to cut), niche subjects (few results)
- **Song choice**: songs with a clear beat (dance, electronic, pop) give better transitions. Ambient music falls back to evenly-spaced timestamps
- **Search queries**: the tool automatically appends `cinematic` and `drone` keywords; no need to include them in your topic
- **Reruns**: if a montage already exists, the tool skips it (useful for `--csv` batch mode)
- **Multiple topics, same song**: the song is cached once in `songs/` and reused

## Generated assets

All generated files (project folders, songs, clips) are gitignored via patterns in `.gitignore`:
- `*__*/` — all project output directories
- `songs/*.wav` — cached song audio
