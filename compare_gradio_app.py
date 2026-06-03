import argparse
import base64
import html
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import huggingface_hub
import numpy as np
import torch


def _patch_huggingface_hub_hffolder() -> None:
    if hasattr(huggingface_hub, "HfFolder"):
        return

    from pathlib import Path

    from huggingface_hub import constants
    from huggingface_hub.utils._auth import _save_token, get_token

    class _CompatHfFolder:
        @staticmethod
        def get_token() -> str | None:
            return get_token()

        @staticmethod
        def save_token(token: str) -> None:
            _save_token(token, "default")
            Path(constants.HF_TOKEN_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(constants.HF_TOKEN_PATH).write_text(token)

        @staticmethod
        def delete_token() -> None:
            Path(constants.HF_TOKEN_PATH).unlink(missing_ok=True)

    huggingface_hub.HfFolder = _CompatHfFolder


_patch_huggingface_hub_hffolder()

import gradio as gr


def _patch_gradio_matplotlib_backend_manager() -> None:
    class _SafeMatplotlibBackendManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    gr.utils.MatplotlibBackendMananger = _SafeMatplotlibBackendManager


_patch_gradio_matplotlib_backend_manager()

try:
    from .compare_motion_utils import PELVIS_INDEX, load_compare_motion_data
except ImportError:
    from compare_motion_utils import PELVIS_INDEX, load_compare_motion_data


REPO_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = REPO_ROOT / "external" / "kimodo"
STMC_ROOT = REPO_ROOT / "external" / "stmc"
KIMODO_SCRIPT = STMC_ROOT / "kimodo_smplx_to_json.py"
STMC_GENERATE_SCRIPT = STMC_ROOT / "generate.py"
APP_CACHE_ROOT = STMC_ROOT / "compare_gradio_cache"
DEFAULT_STMC_RUN_DIR = STMC_ROOT / "pretrained_models" / "mdm-smpl_clip_smplrifke_humanml3d"
DEFAULT_TEXT_FILE = STMC_ROOT / "eval_prompts" / "single_actions_text.txt"
TARGET_FPS = 20.0
SKELETON_JOINT_COUNT = 22
NECK_INDEX = 12
DEFAULT_KIMODO_ENV = "kimodo"
DEFAULT_STMC_ENV = "stmc"

for root in (KIMODO_ROOT, STMC_ROOT):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from kimodo.exports.motion_io import load_kimodo_npz
from kimodo.skeleton.registry import build_skeleton
from kimodo.viz.smplx_skin import SMPLXSkin


def log_status(message: str) -> str:
    timestamp = time.strftime("%H:%M:%S")
    return f"[{timestamp}] {message}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Gradio comparison app for Kimodo vs STMC.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--kimodo_env", default=DEFAULT_KIMODO_ENV, help="Conda env used for Kimodo generation.")
    parser.add_argument("--stmc_env", default=DEFAULT_STMC_ENV, help="Conda env used for STMC generation.")
    parser.add_argument(
        "--stmc_run_dir",
        default=str(DEFAULT_STMC_RUN_DIR),
        help="STMC pretrained run directory passed to generate.py.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose the Gradio app with a public share link.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def run_command(cmd: list[str], cwd: Path | None = None, logs: list[str] | None = None) -> None:
    message = "Running command: " + " ".join(cmd)
    if logs is not None:
        logs.append(log_status(message))
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "prompt"


def load_examples(limit: int = 12) -> list[list]:
    examples = []
    if not DEFAULT_TEXT_FILE.exists():
        return examples
    for raw_line in DEFAULT_TEXT_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("#")]
        if len(parts) < 2:
            continue
        examples.append([parts[0], float(parts[1]), 1234])
        if len(examples) >= limit:
            break
    return examples


APP_ARGS = parse_args()


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
    raise FileNotFoundError("Could not find SMPLX_NEUTRAL.npz for mesh preview.")


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


def encode_array(array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


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
    skeleton = build_skeleton(22)
    edges = []
    for joint_name, parent_name in skeleton.bone_order_names_with_parents:
        if parent_name is None:
            continue
        edges.append([skeleton.bone_index[parent_name], skeleton.bone_index[joint_name]])
    return np.asarray(edges, dtype=np.uint16)


def compute_kimodo_mesh_vertices(joints_pos: np.ndarray, joints_rot: np.ndarray) -> np.ndarray:
    ensure_smplx_asset()
    skeleton = build_skeleton(22)
    skin = SMPLXSkin(skeleton)
    device = skin.skeleton.neutral_joints.device
    joints_pos_t = torch.from_numpy(np.asarray(joints_pos, dtype=np.float32)).to(device)
    joints_rot_t = torch.from_numpy(np.asarray(joints_rot, dtype=np.float32)).to(device)
    with torch.no_grad():
        vertices = skin.skin(joints_rot_t, joints_pos_t, rot_is_global=True).cpu().numpy()
    return np.asarray(vertices, dtype=np.float32)


def generate_prompt_file(prompt: str, duration_s: float, work_dir: Path) -> Path:
    slug = slugify(prompt)[:40]
    prompt_file = work_dir / f"prompt_{slug}.txt"
    prompt_file.write_text(f"{prompt.strip()} # {float(duration_s)}\n")
    return prompt_file


def run_generators(prompt: str, duration_s: float, seed: int, logs: list[str]) -> tuple[Path, Path, Path]:
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_" + slugify(prompt)[:40]
    work_dir = APP_CACHE_ROOT / run_id
    kimodo_dir = work_dir / "kimodo"
    stmc_dir = work_dir / "stmc"
    work_dir.mkdir(parents=True, exist_ok=True)
    kimodo_dir.mkdir(parents=True, exist_ok=True)
    stmc_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = generate_prompt_file(prompt, duration_s, work_dir)

    logs.append(log_status("Starting Kimodo generation-only run"))
    run_command(
        [
            "conda",
            "run",
            "-n",
            APP_ARGS.kimodo_env,
            "python",
            str(KIMODO_SCRIPT),
            "--out_dir",
            str(kimodo_dir),
            "--text_file",
            str(prompt_file),
            "--seed",
            str(seed),
            "--save_motion_npz",
            "--save_amass",
            "--z_up",
            "--no_video",
            "--no_mesh_video",
        ],
        cwd=REPO_ROOT,
        logs=logs,
    )

    logs.append(log_status("Starting STMC generation-only run"))
    run_command(
        [
            "conda",
            "run",
            "-n",
            APP_ARGS.stmc_env,
            "python",
            str(STMC_GENERATE_SCRIPT),
            f"run_dir={Path(APP_ARGS.stmc_run_dir).resolve()}",
            f"timeline={prompt_file}",
            "input_type=text",
            "value_from=smpl",
            "fast=false",
            f"seed={seed}",
            "ckpt=last",
            "guidance=2.5",
            "device=cuda",
            "render_joints=false",
            "render_smpl=false",
        ],
        cwd=STMC_ROOT,
        logs=logs,
    )

    stmc_gen_dir = Path(APP_ARGS.stmc_run_dir).resolve() / "generations" / f"{prompt_file.stem}_last_text_to_motion"
    return work_dir, kimodo_dir / "kimodo_motion.npz", stmc_gen_dir


def package_compare_payload(prompt: str, duration_s: float, kimodo_motion_npz: Path, stmc_gen_dir: Path, logs: list[str]) -> dict:
    target_frames = max(1, int(float(duration_s) * TARGET_FPS))
    stmc_joint_candidates = sorted(stmc_gen_dir.glob("*_text_0.npy"))
    stmc_vert_candidates = sorted(stmc_gen_dir.glob("*_text_0_verts.npy"))
    if not stmc_joint_candidates:
        raise FileNotFoundError(f"Could not find STMC joints output in {stmc_gen_dir}")
    if not stmc_vert_candidates:
        raise FileNotFoundError(f"Could not find STMC verts output in {stmc_gen_dir}")
    stmc_joints_npy = stmc_joint_candidates[0]
    stmc_verts_npy = stmc_vert_candidates[0]
    logs.append(log_status("Loading and synchronizing Kimodo/STMC motion data"))
    data = load_compare_motion_data(
        kimodo_motion_npz=kimodo_motion_npz,
        stmc_joints_npy=stmc_joints_npy,
        stmc_verts_npy=stmc_verts_npy,
        target_frames=target_frames,
    )

    payload = {
        "prompt": prompt,
        "fps": TARGET_FPS,
        "frames": target_frames,
        "neckIndex": NECK_INDEX,
        "pelvisIndex": PELVIS_INDEX,
        "overlayFloorShift": float(data["overlay_floor_shift"]),
        "skeletonEdges": encode_array(data["edges"]),
        "models": {
            "kimodo": {
                "name": "Kimodo-SMPLX-RP",
                "skeleton": encode_array(data["kimodo"]["joints"]),
                "meshVertices": encode_array(data["kimodo"]["vertices"]),
                "meshFaces": encode_array(data["kimodo"]["faces"]),
            },
            "stmc": {
                "name": "STMC MDM-SMPL",
                "skeleton": encode_array(data["stmc"]["joints"]),
                "meshVertices": encode_array(data["stmc"]["vertices"]),
                "meshFaces": encode_array(data["stmc"]["faces"]),
            },
        },
    }
    return payload


def build_viewer_html(payload: dict) -> str:
    viewer_id = f"viewer_{uuid.uuid4().hex}"
    payload_json = json.dumps(payload).replace("</", "<\\/")
    prompt_html = html.escape(payload["prompt"])
    return f"""
<div id="{viewer_id}" class="compare-root">
  <div class="compare-header">
    <div class="compare-prompt">{prompt_html}</div>
    <div class="compare-controls">
      <button id="{viewer_id}_play">Pause</button>
      <label>Frame <input id="{viewer_id}_frame" type="range" min="0" max="{payload["frames"] - 1}" value="0" step="1"></label>
      <label>Distance <input id="{viewer_id}_distance" type="range" min="0.8" max="6.0" value="2.2" step="0.02"></label>
    </div>
  </div>
  <div class="compare-grid">
    <div class="panel"><div class="panel-title">Kimodo Skeleton</div><div id="{viewer_id}_kimodo_skeleton" class="viewer-canvas"></div></div>
    <div class="panel"><div class="panel-title">STMC Skeleton</div><div id="{viewer_id}_stmc_skeleton" class="viewer-canvas"></div></div>
    <div class="panel"><div class="panel-title">Kimodo Mesh</div><div id="{viewer_id}_kimodo_mesh" class="viewer-canvas"></div></div>
    <div class="panel"><div class="panel-title">STMC Mesh</div><div id="{viewer_id}_stmc_mesh" class="viewer-canvas"></div></div>
    <div class="panel overlay-panel">
      <div class="panel-title">Pelvis-Aligned Overlay</div>
      <div class="panel-subtitle">Blue = Kimodo-SMPLX-RP, Amber = STMC MDM-SMPL</div>
      <div id="{viewer_id}_overlay" class="viewer-canvas overlay-canvas"></div>
    </div>
  </div>
  <script type="application/json" id="{viewer_id}_payload">{payload_json}</script>
</div>
<style>
  #{viewer_id}.compare-root {{
    font-family: Georgia, "Times New Roman", serif;
    color: #111;
    background: linear-gradient(180deg, #f8f4ec 0%, #f2eee7 100%);
    border: 1px solid #d9d1c3;
    border-radius: 18px;
    padding: 14px;
  }}
  #{viewer_id} .compare-header {{
    display: flex;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }}
  #{viewer_id} .compare-prompt {{
    font-size: 22px;
    font-weight: 700;
    line-height: 1.2;
  }}
  #{viewer_id} .compare-controls {{
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }}
  #{viewer_id} .compare-controls label {{
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 14px;
  }}
  #{viewer_id} .compare-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(260px, 1fr));
    gap: 14px;
  }}
  #{viewer_id} .panel {{
    background: rgba(255,255,255,0.7);
    border-radius: 14px;
    padding: 10px;
    border: 1px solid rgba(120,100,70,0.18);
  }}
  #{viewer_id} .panel-title {{
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
    text-align: center;
  }}
  #{viewer_id} .panel-subtitle {{
    font-size: 13px;
    text-align: center;
    color: #5f5a50;
    margin-bottom: 8px;
  }}
  #{viewer_id} .viewer-canvas {{
    width: 100%;
    height: 320px;
    border-radius: 10px;
    overflow: hidden;
    background: #ffffff;
  }}
  #{viewer_id} .overlay-panel {{
    grid-column: 1 / -1;
  }}
  #{viewer_id} .overlay-canvas {{
    height: 420px;
  }}
  @media (max-width: 900px) {{
    #{viewer_id} .compare-grid {{
      grid-template-columns: 1fr;
    }}
    #{viewer_id} .overlay-panel {{
      grid-column: auto;
    }}
  }}
</style>
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("{viewer_id}");
  if (!root || root.dataset.initialized === "1") return;
  root.dataset.initialized = "1";

  const payload = JSON.parse(document.getElementById("{viewer_id}_payload").textContent);
  const frameSlider = document.getElementById("{viewer_id}_frame");
  const distanceSlider = document.getElementById("{viewer_id}_distance");
  const playButton = document.getElementById("{viewer_id}_play");

  function decodeArray(spec) {{
    const raw = atob(spec.data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    let array;
    if (spec.dtype === "float32") array = new Float32Array(bytes.buffer);
    else if (spec.dtype === "uint32") array = new Uint32Array(bytes.buffer);
    else if (spec.dtype === "uint16") array = new Uint16Array(bytes.buffer);
    else throw new Error("Unsupported dtype " + spec.dtype);
    return array;
  }}

  const edges = decodeArray(payload.skeletonEdges);
  const models = {{
    kimodo: {{
      name: payload.models.kimodo.name,
      skeleton: decodeArray(payload.models.kimodo.skeleton),
      skeletonShape: payload.models.kimodo.skeleton.shape,
      meshVertices: decodeArray(payload.models.kimodo.meshVertices),
      meshVerticesShape: payload.models.kimodo.meshVertices.shape,
      meshFaces: decodeArray(payload.models.kimodo.meshFaces),
      meshFacesShape: payload.models.kimodo.meshFaces.shape,
    }},
    stmc: {{
      name: payload.models.stmc.name,
      skeleton: decodeArray(payload.models.stmc.skeleton),
      skeletonShape: payload.models.stmc.skeleton.shape,
      meshVertices: decodeArray(payload.models.stmc.meshVertices),
      meshVerticesShape: payload.models.stmc.meshVertices.shape,
      meshFaces: decodeArray(payload.models.stmc.meshFaces),
      meshFacesShape: payload.models.stmc.meshFaces.shape,
    }},
  }};

  function getJoint(model, frame, joint) {{
    const shape = model.skeletonShape;
    const offset = ((frame * shape[1]) + joint) * 3;
    return [
      model.skeleton[offset + 0],
      model.skeleton[offset + 1],
      model.skeleton[offset + 2],
    ];
  }}

  function getNeck(model, frame) {{
    return getJoint(model, frame, payload.neckIndex);
  }}

  function getPelvis(model, frame) {{
    return getJoint(model, frame, payload.pelvisIndex);
  }}

  function getAlignedJoint(model, frame, joint) {{
    const pelvis = getPelvis(model, frame);
    const point = getJoint(model, frame, joint);
    return [
      point[0] - pelvis[0],
      point[1] - pelvis[1],
      point[2] - pelvis[2] + payload.overlayFloorShift,
    ];
  }}

  function createScene(containerId, kind, modelKey, meshColor) {{
    const container = document.getElementById(containerId);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100.0);
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 1.0);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.7);
    directional.position.set(1.5, -1.5, 2.5);
    scene.add(directional);

    let object;
    if (kind === "skeleton") {{
      const positions = new Float32Array(edges.length * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      object = new THREE.LineSegments(
        geometry,
        new THREE.LineBasicMaterial({{ color: 0x1c3d5a, linewidth: 2 }})
      );
    }} else {{
      const model = models[modelKey];
      const vertsPerFrame = model.meshVerticesShape[1];
      const positions = new Float32Array(vertsPerFrame * 3);
      positions.set(model.meshVertices.slice(0, vertsPerFrame * 3));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setIndex(Array.from(model.meshFaces));
      object = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({{
          color: meshColor,
          transparent: true,
          opacity: 0.94,
          side: THREE.DoubleSide,
        }})
      );
    }}
    scene.add(object);

    return {{ container, scene, camera, renderer, object, kind, modelKey }};
  }}

  function createOverlayScene(containerId) {{
    const container = document.getElementById(containerId);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100.0);
    camera.up.set(0, 0, 1);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 1.0);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.7);
    directional.position.set(1.5, -1.5, 2.5);
    scene.add(directional);

    function createSkeletonObject(color) {{
      const positions = new Float32Array(edges.length * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({{ color, linewidth: 2 }});
      const lines = new THREE.LineSegments(geometry, material);
      scene.add(lines);
      return lines;
    }}

    function createMeshObject(modelKey, color) {{
      const model = models[modelKey];
      const vertsPerFrame = model.meshVerticesShape[1];
      const positions = new Float32Array(vertsPerFrame * 3);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setIndex(Array.from(model.meshFaces));
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshBasicMaterial({{
          color,
          transparent: true,
          opacity: 0.34,
          side: THREE.DoubleSide,
        }})
      );
      scene.add(mesh);
      return mesh;
    }}

    return {{
      container,
      scene,
      camera,
      renderer,
      kind: "overlay",
      kimodoSkeleton: createSkeletonObject(0x1b5d9c),
      stmcSkeleton: createSkeletonObject(0xb75a15),
      kimodoMesh: createMeshObject("kimodo", 0x4d8bd0),
      stmcMesh: createMeshObject("stmc", 0xd9823d),
    }};
  }}

  const viewers = [
    createScene("{viewer_id}_kimodo_skeleton", "skeleton", "kimodo", 0x9ec1d9),
    createScene("{viewer_id}_stmc_skeleton", "skeleton", "stmc", 0xe5b08a),
    createScene("{viewer_id}_kimodo_mesh", "mesh", "kimodo", 0x9ec1d9),
    createScene("{viewer_id}_stmc_mesh", "mesh", "stmc", 0xe5b08a),
    createOverlayScene("{viewer_id}_overlay"),
  ];

  function resizeViewer(viewer) {{
    const width = viewer.container.clientWidth;
    const height = viewer.container.clientHeight;
    viewer.camera.aspect = width / Math.max(height, 1);
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.setSize(width, height, false);
  }}

  viewers.forEach(resizeViewer);
  window.addEventListener("resize", () => viewers.forEach(resizeViewer));

  const cameraState = {{
    azimuth: -0.9,
    polar: 1.05,
    radius: parseFloat(distanceSlider.value),
  }};

  function getOverlayTarget(frame) {{
    const kimodoNeck = getAlignedJoint(models.kimodo, frame, payload.neckIndex);
    const stmcNeck = getAlignedJoint(models.stmc, frame, payload.neckIndex);
    return [
      0.5 * (kimodoNeck[0] + stmcNeck[0]),
      0.5 * (kimodoNeck[1] + stmcNeck[1]),
      0.5 * (kimodoNeck[2] + stmcNeck[2]),
    ];
  }}

  function updateCamera(viewer, frame) {{
    const neck = viewer.kind === "overlay" ? getOverlayTarget(frame) : getNeck(models[viewer.modelKey], frame);
    const sp = Math.sin(cameraState.polar);
    const cp = Math.cos(cameraState.polar);
    const ca = Math.cos(cameraState.azimuth);
    const sa = Math.sin(cameraState.azimuth);
    viewer.camera.position.set(
      neck[0] + cameraState.radius * ca * sp,
      neck[1] + cameraState.radius * sa * sp,
      neck[2] + cameraState.radius * cp
    );
    viewer.camera.lookAt(neck[0], neck[1], neck[2]);
  }}

  function updateSkeleton(viewer, frame) {{
    const model = models[viewer.modelKey];
    const pos = viewer.object.geometry.attributes.position.array;
    for (let edgeIdx = 0; edgeIdx < edges.length / 2; edgeIdx++) {{
      const a = edges[edgeIdx * 2 + 0];
      const b = edges[edgeIdx * 2 + 1];
      const ja = getJoint(model, frame, a);
      const jb = getJoint(model, frame, b);
      const offset = edgeIdx * 6;
      pos[offset + 0] = ja[0];
      pos[offset + 1] = ja[1];
      pos[offset + 2] = ja[2];
      pos[offset + 3] = jb[0];
      pos[offset + 4] = jb[1];
      pos[offset + 5] = jb[2];
    }}
    viewer.object.geometry.attributes.position.needsUpdate = true;
  }}

  function updateMesh(viewer, frame) {{
    const model = models[viewer.modelKey];
    const vertsPerFrame = model.meshVerticesShape[1];
    const start = frame * vertsPerFrame * 3;
    const end = start + vertsPerFrame * 3;
    viewer.object.geometry.attributes.position.array.set(model.meshVertices.slice(start, end));
    viewer.object.geometry.attributes.position.needsUpdate = true;
  }}

  function updateOverlaySkeleton(lineObject, modelKey, frame) {{
    const model = models[modelKey];
    const pos = lineObject.geometry.attributes.position.array;
    for (let edgeIdx = 0; edgeIdx < edges.length / 2; edgeIdx++) {{
      const a = edges[edgeIdx * 2 + 0];
      const b = edges[edgeIdx * 2 + 1];
      const ja = getAlignedJoint(model, frame, a);
      const jb = getAlignedJoint(model, frame, b);
      const offset = edgeIdx * 6;
      pos[offset + 0] = ja[0];
      pos[offset + 1] = ja[1];
      pos[offset + 2] = ja[2];
      pos[offset + 3] = jb[0];
      pos[offset + 4] = jb[1];
      pos[offset + 5] = jb[2];
    }}
    lineObject.geometry.attributes.position.needsUpdate = true;
  }}

  function updateOverlayMesh(meshObject, modelKey, frame) {{
    const model = models[modelKey];
    const vertsPerFrame = model.meshVerticesShape[1];
    const pelvis = getPelvis(model, frame);
    const floorShift = payload.overlayFloorShift;
    const srcStart = frame * vertsPerFrame * 3;
    const dst = meshObject.geometry.attributes.position.array;
    for (let i = 0; i < vertsPerFrame; i++) {{
      const src = srcStart + i * 3;
      const dstIdx = i * 3;
      dst[dstIdx + 0] = model.meshVertices[src + 0] - pelvis[0];
      dst[dstIdx + 1] = model.meshVertices[src + 1] - pelvis[1];
      dst[dstIdx + 2] = model.meshVertices[src + 2] - pelvis[2] + floorShift;
    }}
    meshObject.geometry.attributes.position.needsUpdate = true;
  }}

  function updateOverlay(viewer, frame) {{
    updateOverlaySkeleton(viewer.kimodoSkeleton, "kimodo", frame);
    updateOverlaySkeleton(viewer.stmcSkeleton, "stmc", frame);
    updateOverlayMesh(viewer.kimodoMesh, "kimodo", frame);
    updateOverlayMesh(viewer.stmcMesh, "stmc", frame);
  }}

  function renderFrame(frame) {{
    viewers.forEach((viewer) => {{
      if (viewer.kind === "skeleton") updateSkeleton(viewer, frame);
      else if (viewer.kind === "mesh") updateMesh(viewer, frame);
      else updateOverlay(viewer, frame);
      updateCamera(viewer, frame);
      viewer.renderer.render(viewer.scene, viewer.camera);
    }});
  }}

  let currentFrame = 0;
  let playing = true;
  let lastTime = performance.now();
  let frameAccumulator = 0;

  function tick(now) {{
    const dt = (now - lastTime) / 1000.0;
    lastTime = now;
    if (playing) {{
      frameAccumulator += dt * payload.fps;
      if (frameAccumulator >= 1.0) {{
        const step = Math.floor(frameAccumulator);
        frameAccumulator -= step;
        currentFrame = (currentFrame + step) % payload.frames;
        frameSlider.value = String(currentFrame);
      }}
    }}
    renderFrame(currentFrame);
    requestAnimationFrame(tick);
  }}

  playButton.addEventListener("click", () => {{
    playing = !playing;
    playButton.textContent = playing ? "Pause" : "Play";
  }});

  frameSlider.addEventListener("input", () => {{
    currentFrame = parseInt(frameSlider.value, 10);
    renderFrame(currentFrame);
  }});

  distanceSlider.addEventListener("input", () => {{
    cameraState.radius = parseFloat(distanceSlider.value);
    renderFrame(currentFrame);
  }});

  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  function beginDrag(event) {{
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
    event.preventDefault();
  }}
  function onMove(event) {{
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    cameraState.azimuth -= dx * 0.01;
    cameraState.polar = Math.min(Math.PI - 0.1, Math.max(0.1, cameraState.polar + dy * 0.01));
    renderFrame(currentFrame);
  }}
  function endDrag() {{
    dragging = false;
  }}
  viewers.forEach((viewer) => {{
    viewer.renderer.domElement.addEventListener("pointerdown", beginDrag);
    viewer.renderer.domElement.addEventListener("wheel", (event) => {{
      event.preventDefault();
      cameraState.radius = Math.min(6.0, Math.max(0.8, cameraState.radius + event.deltaY * 0.002));
      distanceSlider.value = String(cameraState.radius);
      renderFrame(currentFrame);
    }}, {{ passive: false }});
  }});
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", endDrag);
  window.addEventListener("pointercancel", endDrag);

  renderFrame(0);
  requestAnimationFrame(tick);
}})();
</script>
"""


def generate_compare_view(prompt: str, duration_s: float, seed: int, progress=gr.Progress(track_tqdm=False)):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")

    APP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    logs = [log_status("Preparing comparison run")]
    progress(0.05, desc="Starting generation")

    work_dir, kimodo_motion_npz, stmc_gen_dir = run_generators(prompt, duration_s, seed, logs)
    progress(0.65, desc="Packaging viewer data")

    payload = package_compare_payload(prompt, duration_s, kimodo_motion_npz, stmc_gen_dir, logs)
    viewer_html = build_viewer_html(payload)
    logs.append(log_status(f"Interactive comparison ready at {work_dir}"))
    progress(1.0, desc="Done")
    return viewer_html, "\n".join(logs)


def build_app() -> gr.Blocks:
    examples = load_examples()
    with gr.Blocks(title="Kimodo vs STMC Compare", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Kimodo vs STMC Interactive Compare
            Enter one text prompt, generate both models, and inspect synchronized skeleton, mesh, and pelvis-aligned overlay previews.
            Drag in any viewport to orbit all viewers together. Use the shared `Distance` slider to zoom both models in sync.
            """
        )
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="a person is walking forward", scale=6)
            duration = gr.Slider(label="Duration (seconds)", minimum=2.0, maximum=10.0, step=0.5, value=4.0, scale=2)
            seed = gr.Number(label="Seed", value=1234, precision=0, scale=1)
        generate_btn = gr.Button("Generate Comparison", variant="primary")
        viewer = gr.HTML(label="Interactive Compare")
        logs = gr.Textbox(label="Run Log", lines=14)
        if examples:
            gr.Examples(examples=examples, inputs=[prompt, duration, seed])
        generate_btn.click(generate_compare_view, inputs=[prompt, duration, seed], outputs=[viewer, logs])
    return demo


if __name__ == "__main__":
    build_app().queue().launch(
        server_name=APP_ARGS.host,
        server_port=APP_ARGS.port,
        share=APP_ARGS.share,
    )
