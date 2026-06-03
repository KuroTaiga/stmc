import argparse
from pathlib import Path

try:
    from .compare_motion_utils import TARGET_FPS, load_compare_motion_data, log_status, render_overlay_video
except ImportError:
    from compare_motion_utils import TARGET_FPS, load_compare_motion_data, log_status, render_overlay_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a pelvis-aligned Kimodo vs STMC overlay comparison video.")
    parser.add_argument("--kimodo_motion_npz", required=True)
    parser.add_argument("--stmc_joints_npy", required=True)
    parser.add_argument("--stmc_verts_npy", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--prompt_text", required=True)
    parser.add_argument("--duration_s", type=float, required=True)
    parser.add_argument("--fps", type=float, default=TARGET_FPS)
    parser.add_argument("--figsize", type=float, default=6.0)
    parser.add_argument("--kimodo_label", default="Kimodo-SMPLX-RP")
    parser.add_argument("--stmc_label", default="STMC MDM-SMPL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_frames = max(1, int(float(args.duration_s) * float(args.fps)))
    log_status(f"Loading comparison motions for pelvis-aligned overlay ({target_frames} frames)")
    data = load_compare_motion_data(
        kimodo_motion_npz=Path(args.kimodo_motion_npz),
        stmc_joints_npy=Path(args.stmc_joints_npy),
        stmc_verts_npy=Path(args.stmc_verts_npy),
        target_frames=target_frames,
    )
    log_status(f"Rendering overlay comparison video to {args.out_path}")
    render_overlay_video(
        prompt_text=args.prompt_text,
        kimodo_label=args.kimodo_label,
        stmc_label=args.stmc_label,
        kimodo_joints=data["kimodo"]["aligned_joints"],
        stmc_joints=data["stmc"]["aligned_joints"],
        kimodo_vertices=data["kimodo"]["aligned_vertices"],
        stmc_vertices=data["stmc"]["aligned_vertices"],
        kimodo_faces=data["kimodo"]["faces"],
        stmc_faces=data["stmc"]["faces"],
        edges=data["edges"],
        out_path=Path(args.out_path),
        fps=float(args.fps),
        figsize=float(args.figsize),
    )
    log_status("Overlay comparison render complete")


if __name__ == "__main__":
    main()
