import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = REPO_ROOT / "external" / "kimodo"
STMC_ROOT = REPO_ROOT / "external" / "stmc"

for root in (KIMODO_ROOT, STMC_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from kimodo import load_model
from kimodo.exports.motion_io import load_kimodo_npz, save_kimodo_npz
from kimodo.exports.smplx import kimodo_y_up_to_amass_coord_rotation_matrix
from kimodo.exports.smplx import AMASSConverter, get_amass_parameters
from kimodo.skeleton.registry import build_skeleton
from kimodo.viz.smplx_skin import SMPLXSkin
from src.renderer.matplotlib import MatplotlibRender
from src.stmc import read_timelines


@dataclass(frozen=True)
class PromptSegment:
    text: str
    duration_s: float


@dataclass(frozen=True)
class BatchGenerationJob:
    out_dir: Path
    segments: list[PromptSegment]


def log_status(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SMPL-X motions with Kimodo from direct prompts or STMC prompt files, "
            "then export per-frame JSON plus an optional AMASS NPZ."
        )
    )
    parser.add_argument(
        "--model",
        default="kimodo-smplx-rp",
        help="Kimodo model name. This script expects an SMPL-X model.",
    )
    parser.add_argument(
        "--motion_npz",
        default=None,
        help="Existing Kimodo motion npz to render from. If set, generation is skipped.",
    )
    parser.add_argument(
        "--batch_manifest",
        default=None,
        help=(
            "JSON manifest describing multiple generation jobs. Each job can specify "
            "`out_dir` plus any one of prompts/durations, text_file/text_indices, or "
            "timeline_file/timeline_indices. Loads the Kimodo model once and processes all jobs."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output folder. Single-sample outputs are written directly here; multi-sample outputs use sample_* subfolders.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help='Direct prompt text(s). Example: --prompts "walk forward" "wave right hand"',
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=float,
        default=None,
        help="Duration(s) in seconds for --prompts. Supply one value for all prompts or one per prompt.",
    )
    parser.add_argument(
        "--text_file",
        default=None,
        help="STMC text prompt file such as external/stmc/eval_prompts/single_actions_text.txt",
    )
    parser.add_argument(
        "--text_indices",
        nargs="*",
        type=int,
        default=None,
        help="0-based entries to take from --text_file. Multiple indices are concatenated into one multi-prompt sequence.",
    )
    parser.add_argument(
        "--timeline_file",
        default=None,
        help="STMC timeline prompt file. Only non-overlapping, gap-free timelines are supported.",
    )
    parser.add_argument(
        "--timeline_indices",
        nargs="*",
        type=int,
        default=None,
        help="0-based timeline entries to take from --timeline_file. Selected timelines are concatenated in order.",
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--num_transition_frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cfg_type",
        choices=["nocfg", "regular", "separated"],
        default=None,
        help="Optional Kimodo CFG mode override.",
    )
    parser.add_argument(
        "--cfg_weight",
        nargs="*",
        type=float,
        default=None,
        help="One value for regular CFG, or two values for separated CFG.",
    )
    parser.add_argument(
        "--no_postprocess",
        action="store_true",
        help="Disable Kimodo post-processing.",
    )
    parser.add_argument(
        "--save_amass",
        action="store_true",
        help="Also save the generated motion as amass.npz next to the per-frame JSON.",
    )
    parser.add_argument(
        "--save_motion_npz",
        action="store_true",
        help="Save the raw Kimodo motion dict as kimodo_motion.npz for later render-only use.",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Skip exporting the skeleton animation video.",
    )
    parser.add_argument(
        "--video_name",
        default="motion.mp4",
        help="Filename for the rendered animation video.",
    )
    parser.add_argument(
        "--video_size",
        type=float,
        default=4.0,
        help="Matplotlib figure size used for the exported video.",
    )
    parser.add_argument(
        "--no_mesh_video",
        action="store_true",
        help="Skip exporting the SMPL-X skinned mesh video.",
    )
    parser.add_argument(
        "--mesh_video_name",
        default="motion_mesh.mp4",
        help="Filename for the rendered SMPL-X mesh video.",
    )
    parser.add_argument(
        "--mesh_video_size",
        type=float,
        default=5.0,
        help="Matplotlib figure size used for the exported mesh video.",
    )
    parser.add_argument(
        "--z_up",
        action="store_true",
        help="Export AMASS/JSON motion in standard AMASS Z-up coordinates.",
    )
    parser.add_argument("--width", type=float, default=420.0)
    parser.add_argument("--height", type=float, default=700.0)
    parser.add_argument("--focal", type=float, default=700.0)
    parser.add_argument(
        "--expr_dim",
        type=int,
        default=100,
        help="Number of zero expression coefficients to write into each frame JSON.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only resolve the prompt inputs and print the resulting sequence.",
    )
    return parser.parse_args()


def load_text_entries(path: Path) -> list[PromptSegment]:
    entries: list[PromptSegment] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("#")]
        if len(parts) < 2:
            raise ValueError(f"Invalid text prompt line in {path}: {raw_line!r}")
        entries.append(PromptSegment(text=parts[0], duration_s=float(parts[1])))
    if not entries:
        raise ValueError(f"No prompt entries found in {path}")
    return entries


def select_entries(entries: Sequence[PromptSegment], indices: Sequence[int] | None, label: str) -> list[PromptSegment]:
    if indices is None or len(indices) == 0:
        return list(entries)

    selected: list[PromptSegment] = []
    for index in indices:
        if index < 0 or index >= len(entries):
            raise IndexError(f"{label} index {index} is out of range for {len(entries)} entries")
        selected.append(entries[index])
    return selected


def timeline_to_segments(timeline, path: Path, timeline_index: int) -> list[PromptSegment]:
    timeline = sorted(timeline, key=lambda item: (item.start, item.end))
    if not timeline:
        raise ValueError(f"Timeline {timeline_index} in {path} is empty")

    segments: list[PromptSegment] = []
    current_end = timeline[0].start
    for item in timeline:
        if item.start < current_end - 1e-6:
            raise ValueError(
                f"Timeline {timeline_index} in {path} contains overlapping intervals. "
                "Kimodo multi-prompt mode cannot directly reproduce overlapping STMC body-part timelines."
            )
        if not math.isclose(item.start, current_end, abs_tol=1e-6):
            raise ValueError(
                f"Timeline {timeline_index} in {path} contains a gap before '{item.text}'. "
                "Use direct prompts or a gap-free timeline for this script."
            )
        duration_s = float(item.end - item.start)
        if duration_s <= 0:
            raise ValueError(f"Timeline {timeline_index} in {path} has a non-positive interval for '{item.text}'")
        segments.append(PromptSegment(text=item.text, duration_s=duration_s))
        current_end = item.end
    return segments


def resolve_prompt_segments(args: argparse.Namespace) -> list[PromptSegment]:
    if args.prompts:
        durations = args.durations or [5.0]
        if len(durations) not in (1, len(args.prompts)):
            raise ValueError("For --prompts, provide either one duration or one duration per prompt.")
        if len(durations) == 1:
            durations = durations * len(args.prompts)
        return [PromptSegment(text=text, duration_s=float(duration)) for text, duration in zip(args.prompts, durations)]

    if args.text_file:
        entries = load_text_entries(Path(args.text_file))
        return select_entries(entries, args.text_indices, "text_file")

    if args.timeline_file:
        timelines = read_timelines(args.timeline_file, fps=None)
        indices = args.timeline_indices if args.timeline_indices else list(range(len(timelines)))
        segments: list[PromptSegment] = []
        for index in indices:
            if index < 0 or index >= len(timelines):
                raise IndexError(f"timeline index {index} is out of range for {len(timelines)} timelines")
            segments.extend(timeline_to_segments(timelines[index], Path(args.timeline_file), index))
        return segments

    raise ValueError("Provide one of --prompts, --text_file, or --timeline_file.")


def resolve_batch_job_segments(spec: dict) -> list[PromptSegment]:
    job_args = argparse.Namespace(
        prompts=spec.get("prompts"),
        durations=spec.get("durations"),
        text_file=spec.get("text_file"),
        text_indices=spec.get("text_indices"),
        timeline_file=spec.get("timeline_file"),
        timeline_indices=spec.get("timeline_indices"),
    )
    return resolve_prompt_segments(job_args)


def load_batch_generation_jobs(path: Path) -> list[BatchGenerationJob]:
    payload = json.loads(path.read_text())
    raw_jobs = payload["jobs"] if isinstance(payload, dict) and "jobs" in payload else payload
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError(f"Batch manifest {path} must define a non-empty list of jobs")

    jobs: list[BatchGenerationJob] = []
    for idx, spec in enumerate(raw_jobs):
        if not isinstance(spec, dict):
            raise ValueError(f"Batch manifest job {idx} in {path} must be a JSON object")
        out_dir = spec.get("out_dir")
        if not out_dir:
            raise ValueError(f"Batch manifest job {idx} in {path} is missing `out_dir`")
        segments = resolve_batch_job_segments(spec)
        jobs.append(BatchGenerationJob(out_dir=Path(out_dir), segments=segments))
    return jobs


def build_cfg_kwargs(args: argparse.Namespace) -> dict:
    if args.cfg_type is None and args.cfg_weight is None:
        return {}

    if args.cfg_type == "nocfg":
        if args.cfg_weight:
            raise ValueError("--cfg_weight cannot be used with --cfg_type nocfg")
        return {"cfg_type": "nocfg"}

    if args.cfg_type == "regular":
        if not args.cfg_weight or len(args.cfg_weight) != 1:
            raise ValueError("--cfg_type regular requires exactly one --cfg_weight value")
        return {"cfg_type": "regular", "cfg_weight": float(args.cfg_weight[0])}

    if args.cfg_type == "separated":
        if not args.cfg_weight or len(args.cfg_weight) != 2:
            raise ValueError("--cfg_type separated requires exactly two --cfg_weight values")
        return {"cfg_type": "separated", "cfg_weight": [float(args.cfg_weight[0]), float(args.cfg_weight[1])]}

    if args.cfg_weight is None:
        raise ValueError("--cfg_weight requires --cfg_type regular or separated")

    if len(args.cfg_weight) == 1:
        return {"cfg_type": "regular", "cfg_weight": float(args.cfg_weight[0])}
    if len(args.cfg_weight) == 2:
        return {"cfg_type": "separated", "cfg_weight": [float(args.cfg_weight[0]), float(args.cfg_weight[1])]}
    raise ValueError("--cfg_weight expects one value (regular) or two values (separated)")


def duration_to_frames(duration_s: float, fps: float) -> int:
    return max(1, int(float(duration_s) * float(fps)))


def to_float_list(array_like) -> list[float]:
    return np.asarray(array_like, dtype=np.float32).astype(float).tolist()


def convert_points_for_video(points: np.ndarray) -> np.ndarray:
    y_up_to_z_up = kimodo_y_up_to_amass_coord_rotation_matrix()
    return np.matmul(np.asarray(points, dtype=np.float32), y_up_to_z_up.T)


def convert_joints_for_video(joints: np.ndarray) -> np.ndarray:
    """Convert Kimodo Y-up joints to the Z-up convention expected by STMC's renderer."""
    return convert_points_for_video(joints)


def find_smplx_asset_candidates() -> list[Path]:
    return [
        REPO_ROOT / "external" / "kimodo" / "kimodo" / "assets" / "skeletons" / "smplx22" / "SMPLX_NEUTRAL.npz",
        REPO_ROOT / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz",
        REPO_ROOT / "external" / "LHM_3dnav" / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz",
        REPO_ROOT / "external" / "LHM_pp" / "pretrained_models" / "Damo_XR_Lab" / "LHMPP-Prior" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz",
    ]


def ensure_smplx_skin_asset(skeleton) -> Path:
    dst = Path(skeleton.folder) / "SMPLX_NEUTRAL.npz"
    if dst.exists():
        return dst

    for candidate in find_smplx_asset_candidates():
        if not candidate.exists() or candidate.resolve() == dst.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(candidate, dst)
            log_status(f"Linked SMPL-X skin asset into Kimodo assets: {dst} -> {candidate}")
        except OSError:
            shutil.copy2(candidate, dst)
            log_status(f"Copied SMPL-X skin asset into Kimodo assets: {dst} <- {candidate}")
        return dst

    raise FileNotFoundError(
        "Could not find SMPLX_NEUTRAL.npz for mesh skinning. "
        "Expected one of: " + ", ".join(str(path) for path in find_smplx_asset_candidates())
    )


def save_frame_jsons(
    out_dir: Path,
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    pose_hand: np.ndarray,
    pose_jaw: np.ndarray,
    pose_eye: np.ndarray,
    betas: np.ndarray,
    width: float,
    height: float,
    focal: float,
    expr_dim: int,
) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cx = width / 2.0
    cy = height / 2.0

    left_hand = pose_hand[:, :45].reshape(len(pose_hand), 15, 3)
    right_hand = pose_hand[:, 45:].reshape(len(pose_hand), 15, 3)
    left_eye = pose_eye[:, :3]
    right_eye = pose_eye[:, 3:]

    log_status(f"Writing {trans.shape[0]} per-frame SMPL-X JSON files to {frames_dir}")
    for frame_idx in range(trans.shape[0]):
        frame = {
            "root_pose": to_float_list(root_orient[frame_idx]),
            "body_pose": to_float_list(pose_body[frame_idx].reshape(21, 3)),
            "trans": to_float_list(trans[frame_idx]),
            "betas": to_float_list(betas),
            "jaw_pose": to_float_list(pose_jaw[frame_idx]),
            "leye_pose": to_float_list(left_eye[frame_idx]),
            "reye_pose": to_float_list(right_eye[frame_idx]),
            "lhand_pose": to_float_list(left_hand[frame_idx]),
            "rhand_pose": to_float_list(right_hand[frame_idx]),
            "expr": [0.0] * expr_dim,
            "focal": [float(focal), float(focal)],
            "princpt": [float(cx), float(cy)],
            "img_size_wh": [float(width), float(height)],
        }
        with (frames_dir / f"{frame_idx:06d}.json").open("w") as handle:
            json.dump(frame, handle)


def save_sample_outputs(
    out_dir: Path,
    sample_idx: int,
    sample_output: dict,
    converter: AMASSConverter,
    segments: Sequence[PromptSegment],
    fps: float,
    args: argparse.Namespace,
    resolved_model: str,
) -> None:
    log_status(f"Preparing sample {sample_idx:02d} export data")
    local_rot_mats = np.asarray(sample_output["local_rot_mats"], dtype=np.float32)[None]
    root_positions = np.asarray(sample_output["root_positions"], dtype=np.float32)[None]
    trans, root_orient, pose_body = get_amass_parameters(
        local_rot_mats,
        root_positions,
        converter.skeleton,
        z_up=args.z_up,
    )

    trans = np.asarray(trans[0], dtype=np.float32)
    root_orient = np.asarray(root_orient[0], dtype=np.float32)
    pose_body = np.asarray(pose_body[0], dtype=np.float32)

    num_frames = trans.shape[0]
    pose_jaw = np.repeat(converter.default_frame_params["pose_jaw"][None], num_frames, axis=0).astype(np.float32)
    pose_eye = np.repeat(converter.default_frame_params["pose_eye"][None], num_frames, axis=0).astype(np.float32)
    pose_hand = np.repeat(converter.default_frame_params["pose_hand"][None], num_frames, axis=0).astype(np.float32)
    betas = np.asarray(converter.output_dict_base["betas"], dtype=np.float32)

    if args.save_amass:
        log_status(f"Saving AMASS NPZ for sample {sample_idx:02d} to {out_dir / 'amass.npz'}")
        converter.save_npz(
            trans,
            root_orient,
            pose_body,
            {
                **converter.output_dict_base,
                "pose_jaw": pose_jaw,
                "pose_eye": pose_eye,
                "pose_hand": pose_hand,
                "mocap_time_length": num_frames / fps,
            },
            out_dir / "amass.npz",
        )

    save_frame_jsons(
        out_dir=out_dir,
        trans=trans,
        root_orient=root_orient,
        pose_body=pose_body,
        pose_hand=pose_hand,
        pose_jaw=pose_jaw,
        pose_eye=pose_eye,
        betas=betas,
        width=args.width,
        height=args.height,
        focal=args.focal,
        expr_dim=args.expr_dim,
    )

    meta = {
        "model": resolved_model,
        "fps": float(fps),
        "num_frames": int(num_frames),
        "sample_index": int(sample_idx),
        "z_up": bool(args.z_up),
        "segments": [asdict(segment) for segment in segments],
    }
    with (out_dir / "meta.json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    log_status(f"Saved metadata for sample {sample_idx:02d} to {out_dir / 'meta.json'}")


def save_video(
    out_dir: Path,
    sample_idx: int,
    sample_output: dict,
    fps: float,
    args: argparse.Namespace,
) -> None:
    video_path = out_dir / args.video_name
    log_status(f"Rendering skeleton video for sample {sample_idx:02d} to {video_path}")
    joints = np.asarray(sample_output["posed_joints"], dtype=np.float32)
    joints = convert_joints_for_video(joints)
    renderer = MatplotlibRender(jointstype="smpljoints", fps=fps, figsize=args.video_size, canonicalize=False)
    renderer(
        joints,
        output=str(video_path),
        title="",
        canonicalize=False,
    )
    log_status(f"Finished rendering video for sample {sample_idx:02d}")


def compute_mesh_vertices(sample_idx: int, sample_output: dict, skin: SMPLXSkin) -> np.ndarray:
    log_status(f"Skinning SMPL-X mesh vertices for sample {sample_idx:02d}")
    device = skin.skeleton.neutral_joints.device
    joints_pos = torch.from_numpy(np.asarray(sample_output["posed_joints"], dtype=np.float32)).to(device)
    joints_rot = torch.from_numpy(np.asarray(sample_output["global_rot_mats"], dtype=np.float32)).to(device)
    with torch.no_grad():
        vertices = skin.skin(joints_rot, joints_pos, rot_is_global=True).cpu().numpy()
    log_status(f"Finished mesh skinning for sample {sample_idx:02d} ({vertices.shape[0]} frames)")
    return vertices


def render_mesh_video(
    vertices: np.ndarray,
    faces: np.ndarray,
    root_positions: np.ndarray,
    out_path: Path,
    fps: float,
    figsize: float,
) -> None:
    import av
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = convert_points_for_video(vertices)
    root_positions = convert_points_for_video(root_positions)
    floor = float(vertices[..., 2].min())
    vertices[..., 2] -= floor
    root_positions[..., 2] -= floor

    flat_vertices = vertices.reshape(-1, 3)
    mesh_height = max(1.0, float(flat_vertices[:, 2].max()) * 1.1)
    xy_extent = flat_vertices[:, :2].max(axis=0) - flat_vertices[:, :2].min(axis=0)
    radius = max(1.0, float(max(xy_extent[0], xy_extent[1])) * 0.35)

    fig = plt.figure(figsize=(figsize, figsize), dpi=160)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=20.0, azim=-60.0)
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    writer = None
    stream = None
    try:
        for frame_idx in range(vertices.shape[0]):
            if frame_idx == 0 or frame_idx == vertices.shape[0] - 1 or frame_idx % max(1, vertices.shape[0] // 10) == 0:
                log_status(f"Mesh video render progress: frame {frame_idx + 1}/{vertices.shape[0]}")

            ax.cla()
            ax.set_axis_off()
            ax.view_init(elev=20.0, azim=-60.0)
            ax.grid(False)
            ax.xaxis.pane.set_alpha(0.0)
            ax.yaxis.pane.set_alpha(0.0)
            ax.zaxis.pane.set_alpha(0.0)
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            root_xy = root_positions[frame_idx, :2]
            ax.set_xlim(root_xy[0] - radius, root_xy[0] + radius)
            ax.set_ylim(root_xy[1] - radius, root_xy[1] + radius)
            ax.set_zlim(0.0, mesh_height)

            tris = vertices[frame_idx][faces]
            mesh = Poly3DCollection(
                tris,
                facecolors=(0.72, 0.80, 0.90, 1.0),
                edgecolors=(0.25, 0.25, 0.30, 0.10),
                linewidths=0.03,
            )
            ax.add_collection3d(mesh)

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()

            if writer is None:
                height, width = frame.shape[:2]
                writer = av.open(str(out_path), mode="w")
                stream = writer.add_stream("libx264", rate=max(1, int(round(fps))))
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "medium"}

            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                writer.mux(packet)

        if writer is not None and stream is not None:
            for packet in stream.encode():
                writer.mux(packet)
            writer.close()
    finally:
        plt.close(fig)
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


def save_mesh_video(
    out_dir: Path,
    sample_idx: int,
    sample_output: dict,
    fps: float,
    args: argparse.Namespace,
    skin: SMPLXSkin,
) -> None:
    out_path = out_dir / args.mesh_video_name
    log_status(f"Rendering SMPL-X mesh video for sample {sample_idx:02d} to {out_path}")
    vertices = compute_mesh_vertices(sample_idx, sample_output, skin)
    faces = skin.faces.cpu().numpy()
    root_positions = np.asarray(sample_output["root_positions"], dtype=np.float32)
    render_mesh_video(
        vertices=vertices,
        faces=faces,
        root_positions=root_positions,
        out_path=out_path,
        fps=fps,
        figsize=args.mesh_video_size,
    )
    log_status(f"Finished rendering SMPL-X mesh video for sample {sample_idx:02d}")


def describe_segments(segments: Iterable[PromptSegment]) -> str:
    return "\n".join(f"  - {segment.text!r} for {segment.duration_s:.3f}s" for segment in segments)


def extract_sample_output(output: dict, sample_idx: int) -> dict:
    sample_output = {}
    for key, value in output.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == output["posed_joints"].shape[0]:
            sample_output[key] = array[sample_idx]
        else:
            sample_output[key] = array
    return sample_output


def generate_outputs_for_segments(
    *,
    out_root: Path,
    segments: Sequence[PromptSegment],
    args: argparse.Namespace,
    model,
    resolved_model: str,
    fps: float,
    converter: AMASSConverter,
    smplx_skin: SMPLXSkin | None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    texts = [segment.text for segment in segments]
    num_frames = [duration_to_frames(segment.duration_s, fps) for segment in segments]
    log_status(f"Resolved {len(segments)} prompt segment(s) for {out_root}:")
    print(describe_segments(segments), flush=True)
    log_status(f"Per-segment frame counts: {num_frames}")

    if args.seed is not None:
        from kimodo.tools import seed_everything

        log_status(f"Seeding all RNGs with seed={args.seed} for {out_root}")
        seed_everything(args.seed)

    cfg_kwargs = build_cfg_kwargs(args)
    if cfg_kwargs:
        log_status(f"CFG override for {out_root}: {cfg_kwargs}")

    use_postprocess = not args.no_postprocess
    log_status(
        "Starting motion generation "
        f"for {out_root} (samples={args.num_samples}, diffusion_steps={args.diffusion_steps}, "
        f"transition_frames={args.num_transition_frames}, postprocess={use_postprocess})"
    )
    output = model(
        texts,
        num_frames,
        num_denoising_steps=args.diffusion_steps,
        num_samples=args.num_samples,
        multi_prompt=True,
        num_transition_frames=args.num_transition_frames,
        post_processing=use_postprocess,
        return_numpy=True,
        **cfg_kwargs,
    )
    log_status(f"Motion generation finished for {out_root}")

    generated_samples = int(output["posed_joints"].shape[0])
    log_status(f"Exporting {generated_samples} generated sample(s) into {out_root}")
    for sample_idx in range(generated_samples):
        sample_output = extract_sample_output(output, sample_idx)
        sample_out_dir = out_root if generated_samples == 1 else out_root / f"sample_{sample_idx:02d}"
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        log_status(f"Exporting sample {sample_idx:02d} into {sample_out_dir}")
        if args.save_motion_npz:
            motion_npz_path = sample_out_dir / "kimodo_motion.npz"
            log_status(f"Saving raw Kimodo motion npz for sample {sample_idx:02d} to {motion_npz_path}")
            save_kimodo_npz(str(motion_npz_path), sample_output)
        save_sample_outputs(
            out_dir=sample_out_dir,
            sample_idx=sample_idx,
            sample_output=sample_output,
            converter=converter,
            segments=segments,
            fps=fps,
            args=args,
            resolved_model=resolved_model,
        )
        if not args.no_video:
            save_video(
                out_dir=sample_out_dir,
                sample_idx=sample_idx,
                sample_output=sample_output,
                fps=fps,
                args=args,
            )
        if smplx_skin is not None:
            save_mesh_video(
                out_dir=sample_out_dir,
                sample_idx=sample_idx,
                sample_output=sample_output,
                fps=fps,
                args=args,
                skin=smplx_skin,
            )

    log_status(f"Finished export workflow for {out_root}")


def main() -> None:
    args = parse_args()
    if args.motion_npz is not None:
        if args.out_dir is None:
            raise ValueError("--out_dir is required when using --motion_npz")
        motion_npz = Path(args.motion_npz)
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        log_status(f"Output root ready at {out_root}")
        log_status(f"Loading existing Kimodo motion npz from {motion_npz}")
        sample_output = load_kimodo_npz(str(motion_npz))
        meta_path = motion_npz.with_name("meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            fps = float(meta.get("fps", 30.0))
        else:
            fps = 30.0
        sample_idx = 0
        skeleton = build_skeleton(22)
        if not args.no_video:
            save_video(
                out_dir=out_root,
                sample_idx=sample_idx,
                sample_output=sample_output,
                fps=fps,
                args=args,
            )
        if not args.no_mesh_video:
            asset_path = ensure_smplx_skin_asset(skeleton)
            log_status(f"Using SMPL-X skin asset {asset_path}")
            smplx_skin = SMPLXSkin(skeleton)
            save_mesh_video(
                out_dir=out_root,
                sample_idx=sample_idx,
                sample_output=sample_output,
                fps=fps,
                args=args,
                skin=smplx_skin,
            )
        log_status(f"Render-only workflow completed successfully: {out_root}")
        return

    if args.dry_run:
        if args.batch_manifest is not None:
            jobs = load_batch_generation_jobs(Path(args.batch_manifest))
            log_status(f"Dry run resolved {len(jobs)} batch generation job(s)")
            for job in jobs:
                log_status(f"Dry run job target: {job.out_dir}")
                print(describe_segments(job.segments), flush=True)
        else:
            segments = resolve_prompt_segments(args)
            log_status(f"Resolved {len(segments)} prompt segment(s):")
            print(describe_segments(segments), flush=True)
        log_status("Dry run complete")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log_status(f"Loading Kimodo model '{args.model}' on {device}")
    model, resolved_model = load_model(
        args.model,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
    )

    if "smplx" not in resolved_model:
        raise ValueError(
            f"Resolved model '{resolved_model}' is not an SMPL-X model. "
            "Use an SMPL-X Kimodo checkpoint such as 'kimodo-smplx-rp'."
        )

    fps = float(model.fps)
    log_status(f"Using Kimodo model '{resolved_model}' at {fps:.2f} FPS")

    converter = AMASSConverter(skeleton=model.skeleton, fps=fps)
    smplx_skin = None
    if not args.no_mesh_video:
        asset_path = ensure_smplx_skin_asset(model.skeleton)
        log_status(f"Using SMPL-X skin asset {asset_path}")
        smplx_skin = SMPLXSkin(model.skeleton)

    if args.batch_manifest is not None:
        jobs = load_batch_generation_jobs(Path(args.batch_manifest))
        log_status(f"Processing {len(jobs)} batch generation job(s) with one Kimodo model load")
        for job in jobs:
            generate_outputs_for_segments(
                out_root=job.out_dir,
                segments=job.segments,
                args=args,
                model=model,
                resolved_model=resolved_model,
                fps=fps,
                converter=converter,
                smplx_skin=smplx_skin,
            )
        log_status("All batch exports completed successfully")
        return

    if args.out_dir is None:
        raise ValueError("--out_dir is required unless --batch_manifest is used")
    segments = resolve_prompt_segments(args)
    generate_outputs_for_segments(
        out_root=Path(args.out_dir),
        segments=segments,
        args=args,
        model=model,
        resolved_model=resolved_model,
        fps=fps,
        converter=converter,
        smplx_skin=smplx_skin,
    )
    log_status(f"All exports completed successfully: {args.out_dir}")


if __name__ == "__main__":
    main()
