import shutil
import subprocess
from pathlib import Path

_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

def _is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS

def _is_audio_file(path: str) -> bool:
    return Path(path).suffix.lower() in _AUDIO_EXTS

def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found in PATH. Please install ffmpeg to enable video->audio extraction and frame export."
        )
    return ffmpeg

def export_video_frame(
    video_path: str,
    timestamp_sec: float,
    out_dir: str = "data/processed/extracted_frames",
    prefix: str = "frame",
) -> str:
    """使用 ffmpeg 按时间戳导出视频帧（PNG）。

    这是一个轻量工具：不依赖 AudioProcessor/模型加载。
    """
    ffmpeg = _require_ffmpeg()
    vp = Path(video_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    safe_ts = f"{float(timestamp_sec):.2f}".replace(".", "p")
    out_path = Path(out_dir) / f"{vp.stem}__{prefix}__t{safe_ts}.png"
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        str(float(timestamp_sec)),
        "-i",
        str(vp),
        "-frames:v",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(out_path)

def extract_audio_from_video(
    video_path: str,
    out_dir: str = "data/processed/extracted_audio",
    sample_rate: int = 48000,
) -> str:
    """将视频中的音频抽取为 WAV，返回抽取后的音频路径（带缓存）。"""
    ffmpeg = _require_ffmpeg()
    vp = Path(video_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 输出文件名固定（同名覆盖），保证可重复运行
    out_path = Path(out_dir) / f"{vp.stem}__sr{sample_rate}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(vp),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg audio extraction failed for {video_path}: {stderr}")
    return str(out_path)