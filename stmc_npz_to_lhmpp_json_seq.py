import argparse
import json
import numpy as np
from pathlib import Path


def to_list(x):
    return np.asarray(x, dtype=np.float32).astype(float).tolist()


def convert_one_npz(input_path: Path, out_dir: Path, width, height, focal):
    data = np.load(input_path, allow_pickle=True)

    poses = data["poses"]  # [T, 66]
    trans = data["trans"]  # [T, 3]

    out_dir.mkdir(parents=True, exist_ok=True)

    cx = width / 2.0
    cy = height / 2.0

    for i in range(poses.shape[0]):
        pose = poses[i]

        frame = {
            "root_pose": to_list(pose[:3]),
            "body_pose": to_list(pose[3:66].reshape(21, 3)),
            "trans": to_list(trans[i]),

            "betas": [0.0] * 10,
            "jaw_pose": [0.0, 0.0, 0.0],
            "leye_pose": [0.0, 0.0, 0.0],
            "reye_pose": [0.0, 0.0, 0.0],
            "lhand_pose": [[0.0, 0.0, 0.0] for _ in range(15)],
            "rhand_pose": [[0.0, 0.0, 0.0] for _ in range(15)],
            "expr": [0.0] * 100,

            "focal": [focal, focal],
            "princpt": [cx, cy],
            "img_size_wh": [width, height],
        }

        with open(out_dir / f"{i:06d}.json", "w") as f:
            json.dump(frame, f)

    print(f"[OK] {input_path.name} -> {out_dir} ({poses.shape[0]} frames)")


def main():
    parser = argparse.ArgumentParser()

    # 二选一：单文件 or 文件夹
    parser.add_argument("--input", default=None, help="single STMC *_smpl.npz")
    parser.add_argument("--input_dir", default=None, help="folder containing STMC *.npz files")

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--width", type=float, default=420)
    parser.add_argument("--height", type=float, default=700)
    parser.add_argument("--focal", type=float, default=700)

    args = parser.parse_args()

    out_root = Path(args.out_dir)

    if args.input is not None:
        input_path = Path(args.input)
        convert_one_npz(
            input_path=input_path,
            out_dir=out_root,
            width=args.width,
            height=args.height,
            focal=args.focal,
        )

    elif args.input_dir is not None:
        input_dir = Path(args.input_dir)

        npz_files = sorted(input_dir.glob("*.npz"))

        if len(npz_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {input_dir}")

        for npz_path in npz_files:
            # 每个 npz 输出到一个独立文件夹
            # 例如 xxx_smpl.npz -> out_dir/xxx_smpl/
            one_out_dir = out_root / npz_path.stem

            convert_one_npz(
                input_path=npz_path,
                out_dir=one_out_dir,
                width=args.width,
                height=args.height,
                focal=args.focal,
            )

        print(f"\nDone. Converted {len(npz_files)} npz files to {out_root}")

    else:
        raise ValueError("Please provide either --input or --input_dir")


if __name__ == "__main__":
    main()
