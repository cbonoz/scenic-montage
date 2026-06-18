#!/usr/bin/env -S uv run
import argparse, csv, json, re, subprocess, sys
from pathlib import Path

import librosa
import numpy as np


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def parse_query(query: str) -> tuple[str, str]:
    m = re.match(r'^(.+?), using song (.+)$', query.strip())
    if not m:
        print("Query must be '<topic>, using song <song>'")
        sys.exit(1)
    return m.group(1).strip(), m.group(2).strip()


def check_deps():
    for name in ("ffmpeg", "ffprobe", "yt-dlp"):
        if not subprocess.run(["which", name], capture_output=True).returncode:
            continue
        print(f"Install missing dependency: {name}")
        sys.exit(1)


def get_dur(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", str(path)], capture_output=True, text=True)
    if r.returncode:
        return 0
    return float(json.loads(r.stdout)["format"]["duration"])


def download_song(songs_dir: Path, song_query: str) -> Path:
    slug = slugify(song_query[:60])
    dst = songs_dir / f"{slug}.wav"
    if dst.exists():
        print(f"  song cached: {dst.name}")
        return dst
    songs_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading song: {song_query}")
    r = subprocess.run([
        "yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
        "--concurrent-fragments", "4",
        "-o", str(dst), f"ytsearch1:{song_query}",
    ], capture_output=True, text=True, timeout=120)
    if not dst.exists():
        print(f"  song download failed: {r.stderr[:300]}")
        sys.exit(1)
    return dst


def find_energetic_30s(path: Path) -> float:
    print("  finding best 30s segment...")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    dur = len(y) / sr
    if dur <= 30:
        return 0.0
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_sr = sr / hop
    wins = int(30 * rms_sr)
    if wins >= len(rms):
        return 0.0
    pad_len = (wins - len(rms) % wins) % wins
    cum = np.cumsum(np.pad(rms, (0, pad_len)))
    sums = cum[wins:] - cum[:-wins]
    start = round(np.argmax(sums) / rms_sr)
    return min(start, dur - 30)


def extract_30s(src: Path, dst: Path, start: float):
    if dst.exists():
        return
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-t", "30",
        "-acodec", "pcm_s16le", str(dst),
    ], capture_output=True, check=True)


def detect_beats(path: Path) -> list[float]:
    print("  detecting beats...")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.asarray(tempo).item())
    times = [t for t in librosa.frames_to_time(beats, sr=sr).tolist() if t < 30]
    print(f"  tempo: {bpm:.0f} BPM, {len(times)} beats")
    return times


def pick_transitions(beats: list[float], n: int = 4) -> list[float]:
    if len(beats) < n:
        return [6, 12, 18, 24][:n]
    idx = [int((i + 1) * len(beats) / (n + 1)) for i in range(n)]
    return [beats[i] for i in idx]


def download_clips(project_dir: Path, topic: str, n: int = 5) -> list[Path]:
    d = project_dir / "clips"
    d.mkdir(exist_ok=True)

    seen = sorted(d.glob("*.*"))
    if len(seen) >= n:
        print(f"  {len(seen)} clips cached")
        return seen[:n]

    print(f"  downloading {n} clips: '{topic}'")
    for f in d.glob("*.part"):
        f.unlink(missing_ok=True)

    fmt = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"
    for attempt in range(2):
        subprocess.run([
            "yt-dlp", "-f", fmt, "--concurrent-fragments", "4",
            "--max-downloads", str(n),
            "-o", str(d / "clip_%(autonumber)02d.%(ext)s"),
            f"ytsearch20:{topic}",
        ], capture_output=True, text=True, timeout=300)

        clips = sorted(d.glob("*.*"))
        if len(clips) >= 3:
            break

    clips = sorted(d.glob("*.*"))
    if not clips:
        print("  no videos found for topic")
        sys.exit(1)

    print(f"  got {len(clips)} clips")
    while len(clips) < n:
        clips.append(clips[-1])
    return clips[:n]


def build_filter(clips: list[Path], trans: list[float], durs: list[float]) -> str:
    lines = []
    labels = []
    for i, (cp, dur) in enumerate(zip(clips, durs)):
        vdur = get_dur(cp)
        start = min(3.0, max(0.0, vdur - dur - 1.0))
        lbl = f"v{i}"
        labels.append(lbl)
        lines.append(
            f"[{i}:v]trim=start={start:.1f}:duration={dur:.2f},"
            f"setpts=PTS-STARTPTS,fps=fps=30,"
            f"scale=1920:1080:force_original_aspect_ratio=increase,"
            f"crop=1920:1080,format=yuv420p[{lbl}]"
        )
    out = labels[0]
    for i, (bt, lbl) in enumerate(zip(trans, labels[1:]), 1):
        ol = f"s{i}"
        lines.append(
            f"[{out}][{lbl}]xfade=offset={bt:.2f}:duration=0.5:transition=fade,"
            f"format=yuv420p[{ol}]"
        )
        out = ol
    fade_end = trans[-1] + 0.5 if trans else 29
    lines.append(
        f"[{out}]fade=t=in:st=0:d=1,fade=t=out:st={fade_end}:d=1,"
        f"format=yuv420p[video]"
    )
    lines.append(
        f"[{len(clips)}:a]alimiter=limit=0.9,"
        f"afade=t=in:st=0:d=1.5,afade=t=out:st=28.5:d=1.5[a]"
    )
    return ";\n".join(lines)


def generate(topic: str, song_query: str):
    topic_slug = slugify(topic[:40])
    song_slug = slugify(song_query[:40])
    slug = f"{topic_slug}__{song_slug}"

    project_dir = Path.cwd() / slug
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "clips").mkdir(exist_ok=True)
    songs_dir = Path.cwd() / "songs"

    out_name = f"{slug}.mp4"
    out_path = project_dir / out_name
    if out_path.exists():
        print(f"  already exists: {out_name}")
        return

    print(f"\n  topic: {topic}")
    print(f"  song:  {song_query}")
    print(f"  dir:   {slug}/\n")

    song_path = download_song(songs_dir, song_query)
    start = find_energetic_30s(song_path)
    print(f"  segment: {start}s - {start + 30}s")

    seg_path = project_dir / "song_30s.wav"
    extract_30s(song_path, seg_path, start)

    beats = detect_beats(seg_path)
    trans = pick_transitions(beats)
    print(f"  transitions: {', '.join(f'{t:.1f}s' for t in trans)}")

    clips = download_clips(project_dir, topic)
    durs = []
    prev = 0.0
    for bt in trans:
        durs.append(bt - prev + 0.5)
        prev = bt
    durs.append(max(0.5, 30 - prev))

    fg = build_filter(clips, trans, durs)

    inputs = []
    for c in clips:
        inputs.extend(["-i", str(c)])
    inputs.extend(["-i", str(seg_path)])

    print("  encoding...")
    r = subprocess.run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", fg,
        "-map", "[video]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-threads", "0",
        "-t", "30", str(out_path),
    ], capture_output=True, text=True, timeout=600)

    if r.returncode != 0:
        if "Unknown encoder" in r.stderr:
            print("  retrying with h264_videotoolbox...")
            r = subprocess.run([
                "ffmpeg", "-y", *inputs,
                "-filter_complex", fg,
                "-map", "[video]", "-map", "[a]",
                "-c:v", "h264_videotoolbox", "-q:v", "65",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-t", "30", str(out_path),
            ], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  failed: {r.stderr.strip()[-300:]}")
            return

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  done: {out_name} ({size_mb:.0f} MB)")
    print(f"  {len(clips)} clips, {len(trans)} transitions")


def init_csv():
    combos = [
        ("boston trains", "for the heavens by axjunior"),
        ("tokyo neon streets at night", "midnight city m83"),
        ("northern lights timelapse", "heartbeats josé gonzález"),
        ("surfing hawaii waves", "the less i know the better tame impala"),
        ("safari animals africa", "elephant tame impala"),
        ("new york city sunset skyline", "empire state of mind jay-z"),
        ("snowy mountains skiing", "take me home eddy kim"),
        ("venice canals italy gondola", "cherry wine hozier"),
        ("japanese cherry blossoms spring", "cherry blossom girl"),
        ("iceland waterfalls aurora", "hoppipolla sigur ros"),
    ]
    with open("prompts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "topic", "song", "slug", "status"])
        for i, (topic, song) in enumerate(combos, 1):
            slug = f"{slugify(topic[:40])}__{slugify(song[:40])}"
            w.writerow([i, topic, song, slug, "pending"])
    print("Created prompts.csv with 10 entries")


def batch_csv(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("status", "").strip() == "done":
            continue
        print(f"\n{'='*60}")
        print(f"[{row['id']}] {row['topic']} × {row['song']}")
        generate(row["topic"].strip(), row["song"].strip())
        # mark done
        row["status"] = "done"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'='*60}")
    print("Batch complete — prompts.csv updated")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ~30s montage videos from YouTube clips timed to music."
    )
    parser.add_argument("--topic", "-t", help="Video topic to search")
    parser.add_argument("--song", "-s", help="Song for soundtrack")
    parser.add_argument("query", nargs="?", help="'<topic>, using song <song>'")
    parser.add_argument("--init-csv", action="store_true", help="Create prompts.csv with 10 combos")
    parser.add_argument("--csv", help="Batch-process all entries in prompts.csv")
    args = parser.parse_args()

    check_deps()

    if args.init_csv:
        init_csv()
        return
    if args.csv:
        batch_csv(args.csv)
        return
    if args.topic and args.song:
        generate(args.topic, args.song)
        return
    if args.query:
        topic, song = parse_query(args.query)
        generate(topic, song)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
