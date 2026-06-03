import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = REPO_ROOT / "external" / "kimodo"
STMC_ROOT = REPO_ROOT / "external" / "stmc"
TARGET_FPS = 20.0
SKELETON_JOINT_COUNT = 22
PELVIS_INDEX = 0
NECK_INDEX = 12

KIMODO_MESH_COLOR = (0.26, 0.49, 0.76, 0.34)
STMC_MESH_COLOR = (0.82, 0.47, 0.15, 0.34)
KIMODO_EDGE_COLOR = (0.10, 0.28, 0.50, 0.72)
STMC_EDGE_COLOR = (0.63, 0.31, 0.08, 0.72)
KIMODO_LINE_COLOR = (0.10, 0.28, 0.50, 1.0)
STMC_LINE_COLOR = (0.63, 0.31, 0.08, 1.0)

for root in (KIMODO_ROOT, STMC_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from kimodo.exports.motion_io import load_kimodo_npz
from kimodo.skeleton.registry import build_skeleton
from kimodo.viz.smplx_skin import SMPLXSkin


def log_status(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_smplx_asset() -> Path:
    candidates = [
        REPO_ROOT / "external" / "kimodo" / "kimodo" / "assets" / "skeletons" / "smplx22" / "SMPLX_NEUTRAL.npz",
        REPO_ROOT / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz",
        REPO_ROOT / "external" / "LHM_3dnav" / "pretrained_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz",
        REPO_ROOT / "external" / "LHM_pp" / "pretrained_models" / "Damo_XR_Lab" / "LHMPP-Prior" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz",
    ]
    dst = REPO_ROOT / "external" / "kimodo" / "kimodo" / "assets" / "skeletons" / "smplx22" / "SMPLX_NEUTRAL.npz"
    if dst.exists():
        return dst
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(candidate, dst)
            except OSError:
                import shutil

                shutil.copy2(candidate, dst)
            return dst
    raise FileNotFoundError("Could not find SMPLX_NEUTRAL.npz for comparison rendering.")


def load_smplx_faces() -> np.ndarray:
    asset_path = ensure_smplx_asset()
    with np.load(asset_path, allow_pickle=True) as data:
        return np.asarray(data["f"], dtype=np.uint32)


def load_stmc_faces(num_vertices: int) -> np.ndarray:
    if num_vertices == 10475:
        return load_smplx_faces()
    if num_vertices == 6890:
        return np.asarray(np.load(STMC_ROOT / "src" / "renderer" / "humor_render_tools" / "smplh.faces"), dtype=np.uint32)
    raise ValueError(f"Unsupported STMC mesh vertex count: {num_vertices}")


def kimodo_y_up_to_z_up_matrix() -> np.ndarray:
    y_up_to_z_up = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_z_180 = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rot_z_180 @ y_up_to_z_up).astype(np.float32)


def convert_kimodo_points_to_z_up(points: np.ndarray) -> np.ndarray:
    rot = kimodo_y_up_to_z_up_matrix()
    return np.matmul(np.asarray(points, dtype=np.float32), rot.T)


def resample_frames(array: np.ndarray, target_frames: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    src_frames = array.shape[0]
    if src_frames == target_frames:
        return array
    if src_frames == 1:
        return np.repeat(array, target_frames, axis=0)

    coords = np.linspace(0.0, src_frames - 1, target_frames, dtype=np.float32)
    idx0 = np.floor(coords).astype(np.int64)
    idx1 = np.clip(idx0 + 1, 0, src_frames - 1)
    alpha = (coords - idx0).astype(np.float32)
    flat = array.reshape(src_frames, -1)
    out = (1.0 - alpha[:, None]) * flat[idx0] + alpha[:, None] * flat[idx1]
    return out.reshape((target_frames,) + array.shape[1:]).astype(np.float32)


def compute_skeleton_edges() -> np.ndarray:
    skeleton = build_skeleton(SKELETON_JOINT_COUNT)
    edges = []
    for joint_name, parent_name in skeleton.bone_order_names_with_parents:
        if parent_name is None:
            continue
        edges.append([skeleton.bone_index[parent_name], skeleton.bone_index[joint_name]])
    return np.asarray(edges, dtype=np.uint16)


def compute_kimodo_mesh_vertices(joints_pos: np.ndarray, joints_rot: np.ndarray) -> np.ndarray:
    ensure_smplx_asset()
    skeleton = build_skeleton(SKELETON_JOINT_COUNT)
    skin = SMPLXSkin(skeleton)
    device = skin.skeleton.neutral_joints.device
    joints_pos_t = torch.from_numpy(np.asarray(joints_pos, dtype=np.float32)).to(device)
    joints_rot_t = torch.from_numpy(np.asarray(joints_rot, dtype=np.float32)).to(device)
    with torch.no_grad():
        vertices = skin.skin(joints_rot_t, joints_pos_t, rot_is_global=True).cpu().numpy()
    return np.asarray(vertices, dtype=np.float32)


def align_points_by_pelvis(points: np.ndarray, joints: np.ndarray) -> np.ndarray:
    pelvis = np.asarray(joints[:, PELVIS_INDEX : PELVIS_INDEX + 1, :], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) - pelvis


def compute_overlay_floor_shift(
    kimodo_joints: np.ndarray,
    stmc_joints: np.ndarray,
    kimodo_vertices: np.ndarray,
    stmc_vertices: np.ndarray,
) -> float:
    min_z = min(
        float(kimodo_joints[..., 2].min()),
        float(stmc_joints[..., 2].min()),
        float(kimodo_vertices[..., 2].min()),
        float(stmc_vertices[..., 2].min()),
    )
    return -min_z if min_z < 0.0 else 0.0


def load_compare_motion_data(
    kimodo_motion_npz: Path,
    stmc_joints_npy: Path,
    stmc_verts_npy: Path,
    target_frames: int,
) -> dict:
    kimodo_motion = load_kimodo_npz(str(kimodo_motion_npz))
    kimodo_joints = np.asarray(kimodo_motion["posed_joints"], dtype=np.float32)[:, :SKELETON_JOINT_COUNT]
    kimodo_rots = np.asarray(kimodo_motion["global_rot_mats"], dtype=np.float32)
    kimodo_joints = convert_kimodo_points_to_z_up(kimodo_joints)
    kimodo_vertices = compute_kimodo_mesh_vertices(
        np.asarray(kimodo_motion["posed_joints"], dtype=np.float32),
        kimodo_rots,
    )
    kimodo_vertices = convert_kimodo_points_to_z_up(kimodo_vertices)

    stmc_joints = np.asarray(np.load(stmc_joints_npy), dtype=np.float32)[:, :SKELETON_JOINT_COUNT]
    stmc_vertices = np.asarray(np.load(stmc_verts_npy), dtype=np.float32)

    kimodo_joints = resample_frames(kimodo_joints, target_frames)
    kimodo_vertices = resample_frames(kimodo_vertices, target_frames)
    stmc_joints = resample_frames(stmc_joints, target_frames)
    stmc_vertices = resample_frames(stmc_vertices, target_frames)

    aligned_kimodo_joints = align_points_by_pelvis(kimodo_joints, kimodo_joints)
    aligned_stmc_joints = align_points_by_pelvis(stmc_joints, stmc_joints)
    aligned_kimodo_vertices = align_points_by_pelvis(kimodo_vertices, kimodo_joints)
    aligned_stmc_vertices = align_points_by_pelvis(stmc_vertices, stmc_joints)
    floor_shift = compute_overlay_floor_shift(
        aligned_kimodo_joints,
        aligned_stmc_joints,
        aligned_kimodo_vertices,
        aligned_stmc_vertices,
    )
    if floor_shift:
        aligned_kimodo_joints[..., 2] += floor_shift
        aligned_stmc_joints[..., 2] += floor_shift
        aligned_kimodo_vertices[..., 2] += floor_shift
        aligned_stmc_vertices[..., 2] += floor_shift

    return {
        "edges": compute_skeleton_edges(),
        "kimodo": {
            "joints": kimodo_joints,
            "vertices": kimodo_vertices,
            "aligned_joints": aligned_kimodo_joints,
            "aligned_vertices": aligned_kimodo_vertices,
            "faces": load_smplx_faces(),
        },
        "stmc": {
            "joints": stmc_joints,
            "vertices": stmc_vertices,
            "aligned_joints": aligned_stmc_joints,
            "aligned_vertices": aligned_stmc_vertices,
            "faces": load_stmc_faces(stmc_vertices.shape[1]),
        },
        "overlay_floor_shift": float(floor_shift),
    }


def render_overlay_video(
    prompt_text: str,
    kimodo_label: str,
    stmc_label: str,
    kimodo_joints: np.ndarray,
    stmc_joints: np.ndarray,
    kimodo_vertices: np.ndarray,
    stmc_vertices: np.ndarray,
    kimodo_faces: np.ndarray,
    stmc_faces: np.ndarray,
    edges: np.ndarray,
    out_path: Path,
    fps: float,
    figsize: float = 6.0,
) -> None:
    import av
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat_vertices = np.concatenate([kimodo_vertices.reshape(-1, 3), stmc_vertices.reshape(-1, 3)], axis=0)
    flat_joints = np.concatenate([kimodo_joints.reshape(-1, 3), stmc_joints.reshape(-1, 3)], axis=0)
    combined = np.concatenate([flat_vertices, flat_joints], axis=0)
    xy_extent = combined[:, :2].max(axis=0) - combined[:, :2].min(axis=0)
    radius = max(1.0, float(max(xy_extent[0], xy_extent[1])) * 0.45)
    max_height = max(1.0, float(combined[:, 2].max()) * 1.15)

    fig = plt.figure(figsize=(figsize, figsize), dpi=160)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    writer = None
    stream = None

    try:
        for frame_idx in range(kimodo_joints.shape[0]):
            if frame_idx == 0 or frame_idx == kimodo_joints.shape[0] - 1 or frame_idx % max(1, kimodo_joints.shape[0] // 10) == 0:
                log_status(f"Overlay render progress: frame {frame_idx + 1}/{kimodo_joints.shape[0]}")

            ax.cla()
            ax.view_init(elev=20.0, azim=-60.0)
            ax.set_axis_off()
            ax.grid(False)
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")
            ax.xaxis.pane.set_alpha(0.0)
            ax.yaxis.pane.set_alpha(0.0)
            ax.zaxis.pane.set_alpha(0.0)
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            kimodo_neck = kimodo_joints[frame_idx, NECK_INDEX]
            stmc_neck = stmc_joints[frame_idx, NECK_INDEX]
            neck_center = 0.5 * (kimodo_neck + stmc_neck)

            ax.set_xlim(neck_center[0] - radius, neck_center[0] + radius)
            ax.set_ylim(neck_center[1] - radius, neck_center[1] + radius)
            ax.set_zlim(0.0, max_height)

            kimodo_mesh = Poly3DCollection(
                kimodo_vertices[frame_idx][kimodo_faces],
                facecolors=KIMODO_MESH_COLOR,
                edgecolors=KIMODO_EDGE_COLOR,
                linewidths=0.02,
            )
            stmc_mesh = Poly3DCollection(
                stmc_vertices[frame_idx][stmc_faces],
                facecolors=STMC_MESH_COLOR,
                edgecolors=STMC_EDGE_COLOR,
                linewidths=0.02,
            )
            ax.add_collection3d(kimodo_mesh)
            ax.add_collection3d(stmc_mesh)

            for model_joints, line_color in (
                (kimodo_joints[frame_idx], KIMODO_LINE_COLOR),
                (stmc_joints[frame_idx], STMC_LINE_COLOR),
            ):
                for edge in edges:
                    pts = model_joints[edge]
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=line_color, linewidth=1.6, alpha=0.98)

            ax.text2D(0.02, 0.98, prompt_text, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
            ax.text2D(
                0.02,
                0.93,
                f"{kimodo_label} = blue    {stmc_label} = amber",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
            )

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
