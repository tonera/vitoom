"""
检测视频是否“看起来像 24fps / 是否有明显卡顿（掉帧/变帧率）”。

特点：
- 不依赖项目内部推理代码，可单独对任意 mp4 检测。
- 优先使用 ffprobe（若系统有 ffmpeg/ffprobe）读取准确的容器/流元数据；
- 若安装了 cv2，则用 cv2 抽样读取帧时间戳，统计帧间隔抖动与大间隙（更贴近“卡顿感”）。

示例：
  python test/tools/check_video_fps.py --video /path/to/a.mp4
  python test/tools/check_video_fps.py --video /path/to/a.mp4 --expected-fps 24 --max-frames 2000
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Optional, Tuple


try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import numpy as np  # type: ignore

    NP_AVAILABLE = True
except Exception:
    NP_AVAILABLE = False


def _parse_fraction(fr: str) -> Optional[float]:
    s = (fr or "").strip()
    if not s or s in {"0/0", "N/A"}:
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            num = float(a)
            den = float(b)
            if den == 0:
                return None
            return num / den
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    w = idx - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


@dataclass
class ProbeMeta:
    format_duration_s: Optional[float]
    stream_duration_s: Optional[float]
    nb_frames: Optional[int]
    avg_frame_rate: Optional[float]
    r_frame_rate: Optional[float]
    codec_name: Optional[str]


def _ffprobe_meta(video: Path) -> Optional[ProbeMeta]:
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,r_frame_rate,nb_frames,duration,time_base",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    try:
        obj = json.loads(p.stdout or "{}")
    except Exception:
        return None

    fmt = (obj.get("format") or {}) if isinstance(obj, dict) else {}
    streams = obj.get("streams") if isinstance(obj, dict) else None
    st0 = (streams[0] if isinstance(streams, list) and streams else {}) or {}

    def _to_float(x) -> Optional[float]:
        if x in (None, "N/A", ""):
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _to_int(x) -> Optional[int]:
        if x in (None, "N/A", ""):
            return None
        try:
            return int(float(x))
        except Exception:
            return None

    return ProbeMeta(
        format_duration_s=_to_float(fmt.get("duration")),
        stream_duration_s=_to_float(st0.get("duration")),
        nb_frames=_to_int(st0.get("nb_frames")),
        avg_frame_rate=_parse_fraction(st0.get("avg_frame_rate") or ""),
        r_frame_rate=_parse_fraction(st0.get("r_frame_rate") or ""),
        codec_name=(st0.get("codec_name") if isinstance(st0, dict) else None),
    )


def _sample_timestamps_cv2(video: Path, sample_every: int, max_frames: int) -> Optional[List[float]]:
    if not CV2_AVAILABLE:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    ts: List[float] = []
    i = 0
    kept = 0
    try:
        while kept < max_frames:
            ok, _frame = cap.read()
            if not ok:
                break
            i += 1
            if sample_every > 1 and (i % sample_every) != 0:
                continue
            # CAP_PROP_POS_MSEC：当前位置（毫秒），多数解码器可用；对 VFR 更有意义
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_ms and t_ms > 0:
                ts.append(float(t_ms) / 1000.0)
            kept += 1
    finally:
        cap.release()
    return ts if len(ts) >= 3 else None


def _sample_timestamps_ffprobe(video: Path, sample_every: int, max_frames: int) -> Optional[List[float]]:
    if not shutil.which("ffprobe"):
        return None
    # 注意：show_frames 会输出每帧一行时间戳，可能很大；这里读取到足够数量后就 terminate
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ts: List[float] = []
    i = 0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            s = (line or "").strip()
            if not s:
                continue
            i += 1
            if sample_every > 1 and (i % sample_every) != 0:
                continue
            try:
                ts.append(float(s))
            except Exception:
                continue
            if len(ts) >= max_frames:
                break
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    return ts if len(ts) >= 3 else None


def _analyze_duplicates_cv2(
    video: Path,
    *,
    sample_every: int,
    max_frames: int,
    resize: int,
    dup_threshold: float,
) -> Optional[dict]:
    """
    用 cv2 读取帧并计算相邻帧“像素均值绝对差”(MAD)：
    - MAD 越小，越可能是“重复/几乎不动”的帧
    """
    if not (CV2_AVAILABLE and NP_AVAILABLE):
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None

    def _to_small_gray(frame) -> "np.ndarray":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if resize and resize > 0:
            gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)
        return gray.astype("float32")

    i = 0
    kept = 0
    prev = None
    dists: List[float] = []
    dup_flags: List[bool] = []
    try:
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if sample_every > 1 and (i % sample_every) != 0:
                continue
            cur = _to_small_gray(frame)
            if prev is not None:
                mad = float(np.mean(np.abs(cur - prev)))
                dists.append(mad)
                dup_flags.append(mad <= dup_threshold)
            prev = cur
            kept += 1
    finally:
        cap.release()

    if len(dists) < 5:
        return None

    # 连续重复段统计（dup_flags 对应“当前帧 vs 上一帧”）
    longest_run = 0
    run = 0
    for is_dup in dup_flags:
        if is_dup:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0

    d_sorted = sorted(dists)
    dup_cnt = sum(1 for x in dup_flags if x)
    return {
        "frames_sampled": kept,
        "pairs_analyzed": len(dists),
        "dup_threshold": dup_threshold,
        "dup_count": dup_cnt,
        "dup_ratio": (dup_cnt / len(dists)) if dists else None,
        "longest_dup_run_pairs": longest_run,
        "mad_mean": mean(d_sorted) if d_sorted else None,
        "mad_p50": _percentile(d_sorted, 0.50),
        "mad_p90": _percentile(d_sorted, 0.90),
        "mad_p99": _percentile(d_sorted, 0.99),
    }


def _analyze_intervals(ts: List[float]) -> Tuple[List[float], dict]:
    # 过滤非递增时间戳
    diffs: List[float] = []
    prev = None
    for t in ts:
        if prev is None:
            prev = t
            continue
        dt = t - prev
        prev = t
        if dt > 0:
            diffs.append(dt)
    diffs_sorted = sorted(diffs)
    stats = {
        "interval_count": len(diffs_sorted),
        "dt_mean_s": (mean(diffs_sorted) if diffs_sorted else None),
        "dt_std_s": (pstdev(diffs_sorted) if len(diffs_sorted) >= 2 else None),
        "dt_p50_s": _percentile(diffs_sorted, 0.50),
        "dt_p90_s": _percentile(diffs_sorted, 0.90),
        "dt_p95_s": _percentile(diffs_sorted, 0.95),
        "dt_p99_s": _percentile(diffs_sorted, 0.99),
    }
    return diffs_sorted, stats


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check whether a video is close to expected FPS and if it stutters.")
    p.add_argument("--video", required=True, help="视频文件路径（mp4/webm/mov 等）")
    p.add_argument("--expected-fps", type=float, default=24.0, help="期望帧率（默认 24）")
    p.add_argument("--fps-tol", type=float, default=1.0, help="判定 fps 正常的容差（默认 ±1.0）")
    p.add_argument("--sample-every", type=int, default=1, help="抽样步长：每 N 帧取一次时间戳（默认 1）")
    p.add_argument("--max-frames", type=int, default=2000, help="最多分析多少个抽样帧（默认 2000）")
    p.add_argument(
        "--gap-mult",
        type=float,
        default=1.8,
        help="把 dt > gap_mult*(1/expected_fps) 计为“明显大间隙/疑似卡顿”（默认 1.8）",
    )
    p.add_argument(
        "--check-duplicates",
        action="store_true",
        help="可选：检测疑似重复/几乎不动的帧（需要 cv2 + numpy）；用于解释“明明 24fps 但看起来卡”",
    )
    p.add_argument(
        "--dup-threshold",
        type=float,
        default=1.0,
        help="重复帧判定阈值：相邻帧灰度 MAD <= 阈值视为重复（默认 1.0；越小越严格）",
    )
    p.add_argument(
        "--dup-resize",
        type=int,
        default=64,
        help="重复帧检测时的缩放尺寸（默认 64；越小越快，但越不敏感）",
    )
    return p.parse_args()


def main() -> int:
    args = build_args()
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"[ERR] video not found: {video}")
        return 2

    expected = float(args.expected_fps)
    tol = float(args.fps_tol)
    sample_every = max(1, int(args.sample_every))
    max_frames = max(10, int(args.max_frames))
    gap_mult = float(args.gap_mult)

    meta = _ffprobe_meta(video)
    if meta:
        print("[meta][ffprobe]")
        print(f"  codec_name       : {meta.codec_name}")
        print(f"  format_duration_s: {meta.format_duration_s}")
        print(f"  stream_duration_s: {meta.stream_duration_s}")
        print(f"  nb_frames        : {meta.nb_frames}")
        print(f"  avg_frame_rate   : {meta.avg_frame_rate}")
        print(f"  r_frame_rate     : {meta.r_frame_rate}")
        if meta.nb_frames and meta.format_duration_s and meta.format_duration_s > 0:
            fps_by_count = meta.nb_frames / meta.format_duration_s
            print(f"  fps(nb_frames/duration): {fps_by_count:.3f}")
    else:
        print("[meta] ffprobe not available or failed, metadata will be limited.")

    # 抽样时间戳（优先 cv2，否则 ffprobe show_frames）
    ts = _sample_timestamps_cv2(video, sample_every=sample_every, max_frames=max_frames)
    ts_source = "cv2" if ts else None
    if not ts:
        ts = _sample_timestamps_ffprobe(video, sample_every=sample_every, max_frames=max_frames)
        ts_source = "ffprobe_frames" if ts else None

    if not ts:
        print("[ERR] 无法获取帧时间戳：缺少 cv2 且 ffprobe 不可用/失败。")
        print("      建议：安装 ffmpeg（提供 ffprobe），或安装 opencv-python。")
        return 3

    diffs, stats = _analyze_intervals(ts)
    nominal_dt = 1.0 / expected if expected > 0 else None

    print(f"[timing][{ts_source}] sample_every={sample_every} max_frames={max_frames}")
    print(f"  sampled_timestamps: {len(ts)}")
    print(f"  interval_count    : {stats['interval_count']}")
    if nominal_dt:
        print(f"  nominal_dt_s      : {nominal_dt:.6f}  (expected_fps={expected})")
    for k in ["dt_mean_s", "dt_std_s", "dt_p50_s", "dt_p90_s", "dt_p95_s", "dt_p99_s"]:
        v = stats.get(k)
        if v is not None:
            print(f"  {k:14s}: {float(v):.6f}")

    # 估算“抽样帧率”：1/mean(dt)（注意：sample_every>1 会改变 dt；这里按抽样间隔换算回原 fps）
    fps_est = None
    if stats.get("dt_mean_s") and stats["dt_mean_s"] > 0:
        fps_est = (sample_every / float(stats["dt_mean_s"])) if sample_every > 0 else (1.0 / float(stats["dt_mean_s"]))
        print(f"[estimate] fps_est≈{fps_est:.3f} (from mean dt, adjusted by sample_every)")

    # 卡顿/大间隙计数
    gap_cnt = None
    gap_th = None
    if nominal_dt and diffs:
        gap_th = gap_mult * nominal_dt * sample_every
        gap_cnt = sum(1 for dt in diffs if dt >= gap_th)
        print(f"[stutter] gap_threshold_s={gap_th:.6f}  gap_count={gap_cnt}/{len(diffs)}")

    # 判定：fps 是否接近 expected
    ok = None
    if fps_est is not None:
        ok = abs(fps_est - expected) <= tol
        print(f"[verdict] fps_close_to_expected={ok} (tol=±{tol})")
    else:
        print("[verdict] fps_close_to_expected=unknown (not enough timing info)")

    # 可选：重复/低运动帧检测（解释“时间正常但看起来卡”）
    dup = None
    if bool(getattr(args, "check_duplicates", False)):
        if not CV2_AVAILABLE:
            print("[dup] skipped: cv2 not available")
        elif not NP_AVAILABLE:
            print("[dup] skipped: numpy not available")
        else:
            dup = _analyze_duplicates_cv2(
                video,
                sample_every=sample_every,
                max_frames=max_frames,
                resize=int(args.dup_resize),
                dup_threshold=float(args.dup_threshold),
            )
            if not dup:
                print("[dup] failed: not enough frames or cannot decode")
            else:
                print("[dup][cv2]")
                print(f"  frames_sampled          : {dup['frames_sampled']}")
                print(f"  pairs_analyzed          : {dup['pairs_analyzed']}")
                print(f"  dup_threshold(MAD)      : {dup['dup_threshold']}")
                print(f"  dup_count               : {dup['dup_count']}")
                print(f"  dup_ratio               : {float(dup['dup_ratio']):.4f}" if dup.get("dup_ratio") is not None else "  dup_ratio               : None")
                print(f"  longest_dup_run_pairs   : {dup['longest_dup_run_pairs']}")
                for k in ["mad_mean", "mad_p50", "mad_p90", "mad_p99"]:
                    v = dup.get(k)
                    if v is not None:
                        print(f"  {k:24s}: {float(v):.4f}")

    # 总评：给出一句“正常/不正常”的结论，方便直接判断
    conclusion = "unknown"
    reasons = []
    # 规则（尽量保守）：
    # - fps_est 接近期望，且 gap_cnt 为 0 或很小，且 dt_std 很小 => 时间轴正常
    timing_ok = False
    if (ok is True) and (stats.get("dt_std_s") is not None) and nominal_dt:
        dt_std = float(stats["dt_std_s"])
        # std 超过 10% 帧间隔通常就会明显不稳
        if dt_std <= (nominal_dt * sample_every * 0.10):
            timing_ok = True
        else:
            reasons.append(f"dt_std_s={dt_std:.6f} exceeds 10% of nominal_dt")
    if ok is False:
        reasons.append(f"fps_est={fps_est:.3f} not within ±{tol} of expected_fps={expected}")

    if gap_cnt is not None and diffs:
        gap_ratio = gap_cnt / len(diffs) if len(diffs) > 0 else 0.0
        # 只要出现若干个大间隙就值得提示
        if gap_cnt > 0:
            reasons.append(f"gap_count={gap_cnt} (>{0}) at threshold={gap_th:.6f}")
        # gap 为 0 是强信号
        if gap_cnt == 0 and timing_ok:
            timing_ok = True
    elif ok is True:
        # 没有 diffs/gap 信息，但 fps OK
        timing_ok = True

    if timing_ok:
        conclusion = "正常（时间轴/帧率）"
    elif ok is False or (gap_cnt is not None and gap_cnt > 0):
        conclusion = "不正常（时间轴/帧率）"

    # 若开启重复帧检测：时间轴正常但重复帧高，则给更贴近观感的结论
    if dup and (dup.get("dup_ratio") is not None):
        dup_ratio = float(dup["dup_ratio"])
        longest = int(dup.get("longest_dup_run_pairs") or 0)
        # 经验阈值：>15% 重复 或 连续重复 >= 6（约 0.25s@24fps）都可能“看起来卡”
        if timing_ok and (dup_ratio >= 0.15 or longest >= 6):
            conclusion = "时间轴正常，但可能因重复帧/低运动导致观感卡顿"
            reasons.append(f"dup_ratio={dup_ratio:.3f}, longest_dup_run_pairs={longest}")

    print(f"[conclusion] {conclusion}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

