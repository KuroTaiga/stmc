import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STMC_ROOT = REPO_ROOT / "external" / "stmc"
KIMODO_SCRIPT = REPO_ROOT / "external" / "stmc" / "kimodo_smplx_to_json.py"
STMC_GENERATE_SCRIPT = STMC_ROOT / "generate.py"
STMC_RENDER_SCRIPT = STMC_ROOT / "render.py"
OVERLAY_RENDER_SCRIPT = STMC_ROOT / "render_overlay_compare.py"
DEFAULT_STMC_RUN_DIR = STMC_ROOT / "pretrained_models" / "mdm-smpl_clip_smplrifke_humanml3d"
STMC_JSON_WIDTH = 420.0
STMC_JSON_HEIGHT = 700.0
STMC_JSON_FOCAL = 700.0

try:
    from .stmc_npz_to_lhmpp_json_seq import convert_one_npz
except ImportError:
    from stmc_npz_to_lhmpp_json_seq import convert_one_npz


@dataclass(frozen=True)
class PromptEntry:
    index: int
    text: str
    duration_s: float
    raw_line: str


@dataclass(frozen=True)
class PromptJob:
    entry: PromptEntry
    slug: str
    prompt_file: Path
    prompt_meta_file: Path
    kimodo_dir: Path
    stmc_dir: Path
    compare_skeleton_path: Path
    compare_mesh_path: Path
    compare_overlay_path: Path


def log_status(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Kimodo and STMC on the same prompt entries from an STMC text file, then create "
            "side-by-side labeled comparison videos for skeleton and mesh renderings."
        )
    )
    parser.add_argument(
        "--text_file",
        default=str(STMC_ROOT / "eval_prompts" / "single_actions_text.txt"),
        help="STMC text prompt file in `text # duration` format.",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="0-based prompt indices to compare. Default: all entries in --text_file.",
    )
    parser.add_argument("--out_dir", required=True, help="Folder to store raw generations and combined comparisons.")
    parser.add_argument("--kimodo_env", default="kimodo", help="Conda env name for Kimodo generation.")
    parser.add_argument("--stmc_env", default="stmc", help="Conda env name for STMC generation.")
    parser.add_argument(
        "--stmc_run_dir",
        default=str(DEFAULT_STMC_RUN_DIR),
        help="STMC pretrained run directory passed to generate.py.",
    )
    parser.add_argument("--kimodo_model", default="kimodo-smplx-rp")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--kimodo_diffusion_steps", type=int, default=100)
    parser.add_argument("--kimodo_transition_frames", type=int, default=5)
    parser.add_argument("--stmc_ckpt", default="last")
    parser.add_argument("--stmc_guidance", type=float, default=2.5)
    parser.add_argument("--stmc_device", default="cuda")
    parser.add_argument("--compare_height", type=int, default=720, help="Target height for each side before stacking.")
    parser.add_argument("--compare_fps", type=int, default=30, help="Output FPS for side-by-side videos.")
    parser.add_argument(
        "--overlay_figsize",
        type=float,
        default=6.0,
        help="Matplotlib figure size for the overlay comparison video.",
    )
    parser.add_argument(
        "--header_height",
        type=int,
        default=130,
        help="Total top header height in pixels for prompt title + model labels.",
    )
    parser.add_argument(
        "--prompt_text_size",
        type=int,
        default=34,
        help="Font size for the full-width prompt title at the top.",
    )
    parser.add_argument(
        "--prompt_text_x",
        default="(w-text_w)/2",
        help="FFmpeg drawtext x-expression for the prompt title.",
    )
    parser.add_argument(
        "--prompt_text_y",
        default="14",
        help="FFmpeg drawtext y-expression for the prompt title.",
    )
    parser.add_argument(
        "--model_text_size",
        type=int,
        default=30,
        help="Font size for the per-model labels above each column.",
    )
    parser.add_argument(
        "--left_model_text_x",
        default="w*0.25-text_w/2",
        help="FFmpeg drawtext x-expression for the left model label.",
    )
    parser.add_argument(
        "--right_model_text_x",
        default="w*0.75-text_w/2",
        help="FFmpeg drawtext x-expression for the right model label.",
    )
    parser.add_argument(
        "--model_text_y",
        default="72",
        help="FFmpeg drawtext y-expression for both model labels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing per-prompt outputs for the selected prompts before regenerating.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the selected prompts and planned output locations without running generation.",
    )
    parser.add_argument(
        "--skip_overlay",
        action="store_true",
        help="Skip the slower overlay comparison videos.",
    )
    return parser.parse_args()


def load_prompt_entries(path: Path) -> list[PromptEntry]:
    entries: list[PromptEntry] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("#")]
        if len(parts) < 2:
            raise ValueError(f"Invalid prompt line in {path}: {raw_line!r}")
        entries.append(
            PromptEntry(
                index=len(entries),
                text=parts[0],
                duration_s=float(parts[1]),
                raw_line=f"{parts[0]} # {float(parts[1])}",
            )
        )
    if not entries:
        raise ValueError(f"No prompt entries found in {path}")
    return entries


def select_entries(entries: list[PromptEntry], indices: list[int] | None) -> list[PromptEntry]:
    if not indices:
        return entries

    selected: list[PromptEntry] = []
    for index in indices:
        if index < 0 or index >= len(entries):
            raise IndexError(f"Prompt index {index} is out of range for {len(entries)} entries")
        selected.append(entries[index])
    return selected


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "prompt"


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    log_status("Running command: " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def has_frame_jsons(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.json"))


def export_stmc_frame_jsons(stmc_smpl_npz: Path, stmc_out_dir: Path) -> Path:
    frames_dir = stmc_out_dir / "frames"
    if has_frame_jsons(frames_dir):
        log_status(f"Reusing STMC per-frame JSONs at {frames_dir}")
        return frames_dir

    log_status(f"Exporting STMC SMPL NPZ to LHM++ JSON frames at {frames_dir}")
    convert_one_npz(
        input_path=stmc_smpl_npz,
        out_dir=frames_dir,
        width=STMC_JSON_WIDTH,
        height=STMC_JSON_HEIGHT,
        focal=STMC_JSON_FOCAL,
    )
    return frames_dir


def write_prompt_file(path: Path, entry: PromptEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(entry.raw_line + "\n")


def save_prompt_metadata(path: Path, entry: PromptEntry) -> None:
    payload = {
        "index": entry.index,
        "text": entry.text,
        "duration_s": entry.duration_s,
    }
    path.write_text(json.dumps(payload, indent=2))


def stmc_generation_dir(run_dir: Path, prompt_file: Path, ckpt: str) -> Path:
    stem = prompt_file.stem
    return run_dir / "generations" / f"{stem}_{ckpt}_text_to_motion"


def ffmpeg_escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def combine_videos_side_by_side(
    left_video: Path,
    right_video: Path,
    prompt_text: str,
    left_label: str,
    right_label: str,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top_bar = args.header_height
    prompt_text = ffmpeg_escape_drawtext(prompt_text)
    left_label = ffmpeg_escape_drawtext(left_label)
    right_label = ffmpeg_escape_drawtext(right_label)
    filter_complex = (
        f"[0:v]fps={args.compare_fps},scale=-2:{args.compare_height}[leftbase];"
        f"[1:v]fps={args.compare_fps},scale=-2:{args.compare_height}[rightbase];"
        "[leftbase][rightbase]hstack=inputs=2[stacked];"
        f"[stacked]pad=iw:ih+{top_bar}:0:{top_bar}:color=white,"
        f"drawtext=text='{prompt_text}':fontcolor=black:fontsize={args.prompt_text_size}:"
        f"x={args.prompt_text_x}:y={args.prompt_text_y},"
        f"drawtext=text='{left_label}':fontcolor=black:fontsize={args.model_text_size}:"
        f"x={args.left_model_text_x}:y={args.model_text_y},"
        f"drawtext=text='{right_label}':fontcolor=black:fontsize={args.model_text_size}:"
        f"x={args.right_model_text_x}:y={args.model_text_y}[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left_video),
        "-i",
        str(right_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(out_path),
    ]
    run_command(cmd)


def build_jobs(selected: list[PromptEntry], out_root: Path) -> list[PromptJob]:
    prompts_dir = out_root / "prompts"
    kimodo_root = out_root / "kimodo"
    stmc_root = out_root / "stmc"
    compare_root = out_root / "comparisons"

    for directory in (prompts_dir, kimodo_root, stmc_root, compare_root):
        directory.mkdir(parents=True, exist_ok=True)

    jobs: list[PromptJob] = []
    for entry in selected:
        slug = f"{entry.index:03d}_{slugify(entry.text)}"
        jobs.append(
            PromptJob(
                entry=entry,
                slug=slug,
                prompt_file=prompts_dir / f"{slug}.txt",
                prompt_meta_file=prompts_dir / f"{slug}.json",
                kimodo_dir=kimodo_root / slug,
                stmc_dir=stmc_root / slug,
                compare_skeleton_path=compare_root / f"{slug}_skeleton_compare.mp4",
                compare_mesh_path=compare_root / f"{slug}_mesh_compare.mp4",
                compare_overlay_path=compare_root / f"{slug}_overlay_compare.mp4",
            )
        )
    return jobs


def write_kimodo_batch_manifest(path: Path, jobs: list[PromptJob]) -> None:
    payload = {
        "jobs": [
            {
                "out_dir": str(job.kimodo_dir),
                "text_file": str(job.prompt_file),
            }
            for job in jobs
        ]
    }
    path.write_text(json.dumps(payload, indent=2))


def write_stmc_batch_prompt_file(path: Path, jobs: list[PromptJob]) -> None:
    lines = [job.entry.raw_line for job in jobs]
    path.write_text("\n".join(lines) + "\n")


def run_kimodo_generations(jobs: list[PromptJob], args: argparse.Namespace) -> dict[str, Path]:
    for job in jobs:
        ensure_clean_dir(job.kimodo_dir, args.overwrite)

    pending_jobs = [job for job in jobs if not (job.kimodo_dir / "kimodo_motion.npz").exists()]
    if pending_jobs:
        manifest_path = jobs[0].prompt_file.parent / "kimodo_batch_manifest.json"
        write_kimodo_batch_manifest(manifest_path, pending_jobs)
        cmd = [
            "conda",
            "run",
            "-n",
            args.kimodo_env,
            "python",
            str(KIMODO_SCRIPT),
            "--batch_manifest",
            str(manifest_path),
            "--seed",
            str(args.seed),
            "--model",
            args.kimodo_model,
            "--diffusion_steps",
            str(args.kimodo_diffusion_steps),
            "--num_transition_frames",
            str(args.kimodo_transition_frames),
            "--save_amass",
            "--save_motion_npz",
            "--z_up",
            "--no_video",
            "--no_mesh_video",
        ]
        run_command(cmd, cwd=REPO_ROOT)
    else:
        log_status("Reusing existing Kimodo motion outputs for all selected prompts")

    motion_paths: dict[str, Path] = {}
    for job in jobs:
        motion_npz = job.kimodo_dir / "kimodo_motion.npz"
        if not motion_npz.exists():
            raise FileNotFoundError(f"Missing Kimodo motion npz: {motion_npz}")
        motion_paths[job.slug] = motion_npz
    return motion_paths


def render_kimodo_videos(job: PromptJob, motion_npz: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    skeleton_path = job.kimodo_dir / "kimodo_skeleton.mp4"
    mesh_path = job.kimodo_dir / "kimodo_mesh.mp4"
    if not args.overwrite and skeleton_path.exists() and mesh_path.exists():
        log_status(f"Reusing existing Kimodo videos for prompt {job.entry.index}")
        return skeleton_path, mesh_path

    cmd = [
        "conda",
        "run",
        "-n",
        args.kimodo_env,
        "python",
        str(KIMODO_SCRIPT),
        "--out_dir",
        str(job.kimodo_dir),
        "--motion_npz",
        str(motion_npz),
        "--video_name",
        "kimodo_skeleton.mp4",
        "--mesh_video_name",
        "kimodo_mesh.mp4",
    ]
    run_command(cmd, cwd=REPO_ROOT)
    return skeleton_path, mesh_path


def run_stmc_generations(jobs: list[PromptJob], args: argparse.Namespace) -> dict[str, tuple[Path, Path]]:
    for job in jobs:
        ensure_clean_dir(job.stmc_dir, args.overwrite)

    run_dir = Path(args.stmc_run_dir).resolve()
    pending_jobs = [
        job
        for job in jobs
        if not ((job.stmc_dir / "stmc_joints.npy").exists() and (job.stmc_dir / "stmc_verts.npy").exists())
    ]
    if pending_jobs:
        batch_prompt_file = jobs[0].prompt_file.parent / "stmc_batch_prompts.txt"
        write_stmc_batch_prompt_file(batch_prompt_file, pending_jobs)
        gen_dir = stmc_generation_dir(run_dir, batch_prompt_file, args.stmc_ckpt)
        if gen_dir.exists() and args.overwrite:
            shutil.rmtree(gen_dir)

        cmd = [
            "conda",
            "run",
            "-n",
            args.stmc_env,
            "python",
            str(STMC_GENERATE_SCRIPT),
            f"run_dir={run_dir}",
            f"timeline={batch_prompt_file}",
            "input_type=text",
            "value_from=smpl",
            "fast=false",
            f"seed={args.seed}",
            f"ckpt={args.stmc_ckpt}",
            f"guidance={args.stmc_guidance}",
            f"device={args.stmc_device}",
            "render_joints=false",
            "render_smpl=false",
        ]
        run_command(cmd, cwd=STMC_ROOT)
    else:
        batch_prompt_file = jobs[0].prompt_file.parent / "stmc_batch_prompts.txt"
        gen_dir = stmc_generation_dir(run_dir, batch_prompt_file, args.stmc_ckpt)
        log_status("Reusing existing STMC joint and vertex outputs for all selected prompts")

    asset_paths: dict[str, tuple[Path, Path]] = {}
    for job in jobs:
        joints_dst = job.stmc_dir / "stmc_joints.npy"
        verts_dst = job.stmc_dir / "stmc_verts.npy"
        smpl_dst = job.stmc_dir / "stmc_smpl.npz"
        if pending_jobs and job in pending_jobs:
            idx = pending_jobs.index(job)
            joints_src = gen_dir / f"{batch_prompt_file.stem}_text_{idx}.npy"
            verts_src = gen_dir / f"{batch_prompt_file.stem}_text_{idx}_verts.npy"
            smpl_src = gen_dir / f"{batch_prompt_file.stem}_text_{idx}_smpl.npz"
            if not joints_src.exists():
                raise FileNotFoundError(f"Missing STMC joints output: {joints_src}")
            if not verts_src.exists():
                raise FileNotFoundError(f"Missing STMC verts output: {verts_src}")
            if not smpl_src.exists():
                raise FileNotFoundError(f"Missing STMC SMPL output needed for frames/*.json: {smpl_src}")
            shutil.copy2(joints_src, joints_dst)
            shutil.copy2(verts_src, verts_dst)
            shutil.copy2(smpl_src, smpl_dst)
        elif not joints_dst.exists() or not verts_dst.exists():
            raise FileNotFoundError(
                f"Missing existing STMC assets for prompt {job.entry.index}: {joints_dst}, {verts_dst}"
            )
        if smpl_dst.exists():
            export_stmc_frame_jsons(smpl_dst, job.stmc_dir)
        elif not has_frame_jsons(job.stmc_dir / "frames"):
            raise FileNotFoundError(f"Missing STMC SMPL NPZ needed for frames/*.json: {smpl_dst}")
        asset_paths[job.slug] = (joints_dst, verts_dst)
    return asset_paths


def render_stmc_videos(job: PromptJob, joints_npy: Path, verts_npy: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    skeleton_path = job.stmc_dir / "stmc_skeleton.mp4"
    mesh_path = job.stmc_dir / "stmc_mesh.mp4"

    if not args.overwrite and skeleton_path.exists():
        log_status(f"Reusing existing STMC skeleton video for prompt {job.entry.index}")
    else:
        cmd_joints = [
            "conda",
            "run",
            "-n",
            args.stmc_env,
            "python",
            str(STMC_RENDER_SCRIPT),
            f"path={joints_npy}",
            f"out_path={skeleton_path}",
            "fps=20.0",
        ]
        run_command(cmd_joints, cwd=STMC_ROOT)

    if not args.overwrite and mesh_path.exists():
        log_status(f"Reusing existing STMC mesh video for prompt {job.entry.index}")
    else:
        cmd_mesh = [
            "conda",
            "run",
            "-n",
            args.stmc_env,
            "python",
            str(STMC_RENDER_SCRIPT),
            f"path={verts_npy}",
            f"out_path={mesh_path}",
            "fps=20.0",
        ]
        run_command(cmd_mesh, cwd=STMC_ROOT)
    return skeleton_path, mesh_path


def render_overlay_compare_video(job: PromptJob, kimodo_motion_npz: Path, joints_npy: Path, verts_npy: Path, args: argparse.Namespace) -> Path:
    if not args.overwrite and job.compare_overlay_path.exists():
        log_status(f"Reusing existing overlay comparison video for prompt {job.entry.index}")
        return job.compare_overlay_path

    cmd = [
        "conda",
        "run",
        "-n",
        args.kimodo_env,
        "python",
        str(OVERLAY_RENDER_SCRIPT),
        "--kimodo_motion_npz",
        str(kimodo_motion_npz),
        "--stmc_joints_npy",
        str(joints_npy),
        "--stmc_verts_npy",
        str(verts_npy),
        "--out_path",
        str(job.compare_overlay_path),
        "--prompt_text",
        job.entry.text,
        "--duration_s",
        str(job.entry.duration_s),
        "--fps",
        str(args.compare_fps),
        "--figsize",
        str(args.overlay_figsize),
        "--kimodo_label",
        "Kimodo-SMPLX-RP",
        "--stmc_label",
        "STMC MDM-SMPL",
    ]
    run_command(cmd, cwd=REPO_ROOT)
    return job.compare_overlay_path


def main() -> None:
    args = parse_args()
    text_file = Path(args.text_file).resolve()
    out_root = Path(args.out_dir).resolve()

    log_status(f"Loading prompt file {text_file}")
    entries = load_prompt_entries(text_file)
    selected = select_entries(entries, args.indices)
    log_status(f"Selected {len(selected)} prompt(s) for comparison")
    jobs = build_jobs(selected, out_root)

    for job in jobs:
        write_prompt_file(job.prompt_file, job.entry)
        save_prompt_metadata(job.prompt_meta_file, job.entry)
        log_status(f"Comparing prompt {job.entry.index}: {job.entry.text!r} ({job.entry.duration_s:.2f}s)")
        if args.dry_run:
            log_status(f"Dry run: prompt file would be written to {job.prompt_file}")
            log_status(f"Dry run: Kimodo output dir would be {job.kimodo_dir}")
            log_status(f"Dry run: STMC output dir would be {job.stmc_dir}")
            log_status(f"Dry run: skeleton comparison would be {job.compare_skeleton_path}")
            log_status(f"Dry run: mesh comparison would be {job.compare_mesh_path}")
            log_status(f"Dry run: overlay comparison would be {job.compare_overlay_path}")

    if args.dry_run:
        log_status("Dry run complete")
        return

    log_status("Phase 1/4: running all Kimodo generations")
    kimodo_motion_paths = run_kimodo_generations(jobs, args)

    log_status("Phase 2/4: running all STMC generations")
    stmc_asset_paths = run_stmc_generations(jobs, args)

    log_status("Phase 3/4: rendering all individual videos")
    kimodo_video_paths: dict[str, tuple[Path, Path]] = {}
    log_status("Phase 3a: rendering all Kimodo videos")
    for job in jobs:
        kimodo_video_paths[job.slug] = render_kimodo_videos(job, kimodo_motion_paths[job.slug], args)

    stmc_video_paths: dict[str, tuple[Path, Path]] = {}
    log_status("Phase 3b: rendering all STMC videos")
    for job in jobs:
        joints_npy, verts_npy = stmc_asset_paths[job.slug]
        stmc_video_paths[job.slug] = render_stmc_videos(job, joints_npy, verts_npy, args)

    log_status("Phase 4/4: building side-by-side comparison videos")
    for job in jobs:
        kimodo_skeleton, kimodo_mesh = kimodo_video_paths[job.slug]
        stmc_skeleton, stmc_mesh = stmc_video_paths[job.slug]
        joints_npy, verts_npy = stmc_asset_paths[job.slug]

        if not args.overwrite and job.compare_skeleton_path.exists():
            log_status(f"Reusing existing labeled skeleton comparison video for prompt {job.entry.index}")
        else:
            log_status(f"Building labeled skeleton comparison video for prompt {job.entry.index}")
            combine_videos_side_by_side(
                left_video=kimodo_skeleton,
                right_video=stmc_skeleton,
                prompt_text=job.entry.text,
                left_label="Kimodo-SMPLX-RP",
                right_label="STMC MDM-SMPL",
                out_path=job.compare_skeleton_path,
                args=args,
            )

        if not args.overwrite and job.compare_mesh_path.exists():
            log_status(f"Reusing existing labeled mesh comparison video for prompt {job.entry.index}")
        else:
            log_status(f"Building labeled mesh comparison video for prompt {job.entry.index}")
            combine_videos_side_by_side(
                left_video=kimodo_mesh,
                right_video=stmc_mesh,
                prompt_text=job.entry.text,
                left_label="Kimodo-SMPLX-RP",
                right_label="STMC MDM-SMPL",
                out_path=job.compare_mesh_path,
                args=args,
            )

        if args.skip_overlay:
            log_status(f"Skipping overlay comparison video for prompt {job.entry.index}")
        else:
            log_status(f"Building overlay comparison video for prompt {job.entry.index}")
            render_overlay_compare_video(
                job=job,
                kimodo_motion_npz=kimodo_motion_paths[job.slug],
                joints_npy=joints_npy,
                verts_npy=verts_npy,
                args=args,
            )

    log_status(f"All comparisons completed successfully: {out_root / 'comparisons'}")


if __name__ == "__main__":
    main()
