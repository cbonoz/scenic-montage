#!/usr/bin/env -S uv run
import argparse, csv, json, re, subprocess, sys
from pathlib import Path

import librosa
import numpy as np


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


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


def probe_video(path: Path) -> dict:
    r = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name",
        str(path),
    ], capture_output=True, text=True)
    if r.returncode:
        return {"width": 0, "height": 0}
    streams = json.loads(r.stdout).get("streams", [])
    return streams[0] if streams else {"width": 0, "height": 0}


def clip_quality_score(info: dict) -> float:
    w, h = info.get("width", 0) or 0, info.get("height", 0) or 0
    if w == 0 or h == 0:
        return -1
    aspect = w / h
    if aspect < 1.2 or aspect > 2.0:
        return -2
    aspect_score = -abs(aspect - 16 / 9) * 10
    res = min(w, h)
    if res >= 1080:
        res_score = 100
    elif res >= 720:
        res_score = 70
    elif res >= 480:
        res_score = 40
    else:
        return -3
    return res_score + aspect_score


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


def find_energetic_segment(path: Path, duration: float = 60.0) -> float:
    print(f"  finding best {duration}s segment...")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    dur = len(y) / sr
    if dur <= duration:
        return 0.0
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_sr = sr / hop
    wins = int(duration * rms_sr)
    if wins >= len(rms):
        return 0.0
    pad_len = (wins - len(rms) % wins) % wins
    cum = np.cumsum(np.pad(rms, (0, pad_len)))
    sums = cum[wins:] - cum[:-wins]
    start = round(np.argmax(sums) / rms_sr)
    return min(start, dur - duration)


def extract_segment(src: Path, dst: Path, start: float, duration: float = 60.0):
    if dst.exists():
        return
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-t", str(duration),
        "-acodec", "pcm_s16le", str(dst),
    ], capture_output=True, check=True)


def detect_beats(path: Path, duration: float = 60.0) -> list[float]:
    print("  detecting beats...")
    y, sr = librosa.load(str(path), sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.asarray(tempo).item())
    times = [t for t in librosa.frames_to_time(beats, sr=sr).tolist() if t < duration]
    print(f"  tempo: {bpm:.0f} BPM, {len(times)} beats")
    return times


def pick_transitions(beats: list[float], n: int = 7, duration: float = 60.0) -> list[float]:
    if len(beats) < n:
        step = duration / (n + 1)
        return [round(step * (i + 1), 1) for i in range(n)]
    idx = [int((i + 1) * len(beats) / (n + 1)) for i in range(n)]
    return [beats[i] for i in idx]


def flat_search(topic: str, n: int = 20) -> list[dict]:
    queries = [f"{topic} cinematic", f"{topic} drone", f"{topic}"]
    seen_ids = set()
    results = []
    for q in queries:
        r = subprocess.run([
            "yt-dlp", "--flat-playlist", "--dump-json", "--no-warnings",
            "--playlist-items", f"1-{n}", f"ytsearch{n}:{q}",
        ], capture_output=True, text=True, timeout=30)
        if r.returncode:
            continue
        for line in r.stdout.strip().split("\n"):
            if not line:
                continue
            d = json.loads(line)
            if d.get("id") in seen_ids:
                continue
            seen_ids.add(d["id"])
            results.append(d)
    return results


def deep_probe(videos: list[dict]) -> list[dict]:
    urls = [f"https://www.youtube.com/watch?v={v['id']}" for v in videos]
    r = subprocess.run([
        "yt-dlp", "--dump-json", "--no-download", "--no-warnings",
        *urls,
    ], capture_output=True, text=True, timeout=120)
    if r.returncode:
        return []
    enriched = {}
    for line in r.stdout.strip().split("\n"):
        if not line:
            continue
        d = json.loads(line)
        enriched[d["id"]] = d
    out = []
    for v in videos:
        if v["id"] in enriched:
            out.append(enriched[v["id"]])
        else:
            out.append(v)
    return out


def rank_candidate(d: dict, topic_keywords: set | None = None) -> float:
    dur = d.get("duration") or 0
    views = d.get("view_count") or 0
    w = d.get("width") or 0
    h = d.get("height") or 0

    if dur < 20 or dur > 300:
        return -1

    if w and h:
        aspect = w / h
        if aspect < 1.2:
            return -1
        res = min(w, h)
        if res >= 1080:
            res_score = 100
        elif res >= 720:
            res_score = 70
        elif res >= 480:
            res_score = 40
        else:
            return -1
        aspect_penalty = abs(aspect - 16 / 9) * 15
        over2k = 20 if (w > 2560 or h > 1440) else 0
    else:
        res_score = 50
        aspect_penalty = 0
        over2k = 0

    dur_penalty = max(0, dur - 90) * 0.3
    view_bonus = min(views / 100_000, 10)

    title_bonus = 0
    if topic_keywords and d.get("title"):
        low_title = d["title"].lower()
        matches = sum(1 for kw in topic_keywords if kw in low_title)
        if matches > 0:
            title_bonus = matches * 15
        else:
            title_bonus = -30

    return res_score - aspect_penalty + view_bonus - dur_penalty - over2k + title_bonus


def download_clips(project_dir: Path, topic: str, n: int = 8) -> list[Path]:
    d = project_dir / "clips"
    d.mkdir(exist_ok=True)

    topic_keywords = {kw for kw in topic.lower().split() if len(kw) > 2}

    cached = sorted(d.glob("*.*"))
    if len(cached) >= n:
        with_quality = [(clip_quality_score(probe_video(p)), p) for p in cached]
        good = sorted([p for s, p in with_quality if s > 0], key=lambda p: -clip_quality_score(probe_video(p)))
        if len(good) >= n:
            print(f"  using {len(good)} cached clips")
            return good[:n]

    for f in d.glob("*.part"):
        f.unlink(missing_ok=True)

    print(f"  scanning YouTube for best clips: '{topic}'")
    flat = flat_search(topic)
    pre_ranked = sorted(
        [c for c in flat if rank_candidate(c, topic_keywords) >= 0],
        key=lambda c: rank_candidate(c, topic_keywords), reverse=True,
    )
    print(f"  found {len(flat)} results, {len(pre_ranked)} usable (probing top {min(10, len(pre_ranked))})")
    candidates = deep_probe(pre_ranked[:10]) if pre_ranked else []
    ranked = sorted(
        [c for c in candidates if rank_candidate(c, topic_keywords) >= 0],
        key=lambda c: rank_candidate(c, topic_keywords), reverse=True,
    )

    if not ranked:
        print("  no usable videos found (all too short, vertical, or low-res)")
        sys.exit(1)

    want = min(n + 3, len(ranked))
    urls = [c.get("webpage_url") or f"https://www.youtube.com/watch?v={c['id']}" for c in ranked[:want]]

    fmt = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]"
    print(f"  downloading top {len(urls)}...")
    subprocess.run([
        "yt-dlp", "-f", fmt, "--concurrent-fragments", "4",
        "--max-filesize", "200M",
        "--match-filter", "duration < 300",
        "--no-warnings",
        "-o", str(d / "clip_%(autonumber)02d.%(ext)s"),
        *urls,
    ], capture_output=True, text=True, timeout=600)

    clips = sorted(d.glob("*.*"))
    with_quality = [(clip_quality_score(probe_video(p)), p) for p in clips]
    good = sorted([p for s, p in with_quality if s >= 0], key=lambda p: -clip_quality_score(probe_video(p)))

    if len(good) < n:
        print(f"  only {len(good)} landscape clips, padding with best available")
        good = sorted(clips, key=lambda p: -clip_quality_score(probe_video(p)))

    print(f"  using {min(n, len(good))} clips")
    while len(good) < n:
        good.append(good[-1])
    return good[:n]


def build_filter(clips: list[Path], trans: list[float], durs: list[float], duration: float = 60.0) -> str:
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
    fade_end = duration - 0.5
    lines.append(
        f"[{out}]fade=t=in:st=0:d=1,fade=t=out:st={fade_end}:d=0.5,"
        f"format=yuv420p[video]"
    )
    lines.append(
        f"[{len(clips)}:a]alimiter=limit=0.9,"
        f"afade=t=in:st=0:d=1.5,afade=t=out:st={duration - 0.75}:d=0.75[a]"
    )
    return ";\n".join(lines)


def generate(topic: str, song_query: str, duration: float = 60.0):
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
    start = find_energetic_segment(song_path, duration)
    print(f"  segment: {start}s - {start + duration}s")

    seg_path = project_dir / "song_30s.wav"
    extract_segment(song_path, seg_path, start, duration)

    beats = detect_beats(seg_path, duration)

    n_clips = max(5, int(duration // 7))
    n_trans = n_clips - 1
    trans = pick_transitions(beats, n_trans, duration)
    print(f"  transitions ({n_trans}): {', '.join(f'{t:.1f}s' for t in trans)}")

    clips = download_clips(project_dir, topic, n_clips)
    durs = []
    prev = 0.0
    for bt in trans:
        durs.append(bt - prev + 0.5)
        prev = bt
    durs.append(max(0.5, duration - prev))

    fg = build_filter(clips, trans, durs, duration)

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
        "-t", str(duration), str(out_path),
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
                "-t", str(duration), str(out_path),
            ], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  failed: {r.stderr.strip()[-300:]}")
            return

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  done: {out_name} ({size_mb:.0f} MB)")
    print(f"  {len(clips)} clips, {len(trans)} transitions, {duration:.0f}s")


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
        row["status"] = "done"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'='*60}")
    print("Batch complete — prompts.csv updated")


def main():
    parser = argparse.ArgumentParser(
        description="Generate montage videos from YouTube clips timed to music."
    )
    parser.add_argument("--topic", "-t", help="Video topic to search")
    parser.add_argument("--song", "-s", help="Song for soundtrack")
    parser.add_argument("--duration", "-d", type=float, default=60.0,
                        help="Target duration in seconds (default: 60)")
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
        generate(args.topic, args.song, args.duration)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
