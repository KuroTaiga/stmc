from typing import Optional

import os
import warnings

import torch
from torch import nn

from einops import rearrange, repeat
from torch import Tensor
import numpy as np

from functools import reduce
import operator

from src.tools.smplx_hack import SMPLHLayer, SMPLLayer
from src.tools.geometry import to_matrix

# Extract a 24-joint set used throughout STMC:
# pelvis + 21 body joints + one representative left/right hand joint.
# The hand-joint offsets depend on the underlying body model.
# fmt: off
JOINTS_EXTRACTOR = {
    "smplh": {
        "smpljoints": np.array(
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 37
            ]
        )
    },
    "smplx": {
        "smpljoints": np.array(
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 25, 40
            ]
        )
    },
}
# fmt: on


def call_by_chunks(
    function,
    nelements: int,
    batch_size: int,
    parameters_dict_to_chunk: dict,
    other_parameters: dict = {},
):
    for chunk in range(int((nelements - 1) / batch_size) + 1):
        params = other_parameters.copy()
        cslice = slice(chunk * batch_size, (chunk + 1) * batch_size)

        for key, val in parameters_dict_to_chunk.items():
            params[key] = val[cslice] if val is not None else val
        yield function(**params)


def extract_data(smpl_data, jointstype, model_type="smplh"):
    assert jointstype in ["smpljoints", "vertices", "both"]

    if jointstype == "vertices":
        return smpl_data.vertices

    joints = smpl_data.joints
    extractor_family = "smplx" if model_type == "smplx" else "smplh"
    if jointstype == "both":
        extractor = JOINTS_EXTRACTOR[extractor_family]["smpljoints"]
    else:
        extractor = JOINTS_EXTRACTOR[extractor_family][jointstype]
    joints = joints[..., extractor, :]

    if jointstype == "both":
        return smpl_data.vertices, joints
    return joints


class SMPLH(nn.Module):
    def __init__(
        self,
        path: str,
        jointstype: str = "smpljoints",
        input_pose_rep: str = "matrix",
        batch_size: int = 512,
        num_betas: int = 16,
        gender: str = "neutral",
        **kwargs
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.input_pose_rep = input_pose_rep
        self.jointstype = jointstype
        self.model_path, self.model_type = find_body_model_path(path)

        if self.model_type == "smplh":
            self.smplh = SMPLHLayer(
                self.model_path, ext="npz", gender=gender, num_betas=num_betas
            ).eval()
        elif self.model_type == "smpl":
            self.smplh = SMPLLayer(
                self.model_path, gender=gender, num_betas=num_betas
            ).eval()
        elif self.model_type == "smplx":
            from smplx import SMPLXLayer as ExternalSMPLXLayer

            self.smplh = ExternalSMPLXLayer(
                self.model_path, ext="npz", gender=gender, num_betas=num_betas
            ).eval()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.faces = self.smplh.faces

        self.eval()

        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    def forward(
        self,
        poses,
        trans,
        betas: Optional = None,
        jointstype: Optional[str] = None,
        input_pose_rep: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        # Take values from init if not specified there
        jointstype = self.jointstype if jointstype is None else jointstype
        batch_size = self.batch_size if batch_size is None else batch_size
        input_pose_rep = (
            self.input_pose_rep if input_pose_rep is None else input_pose_rep
        )

        needs_to_squeeze = False
        if len(trans.shape) == 2:
            needs_to_squeeze = True
            poses = poses[None]
            trans = trans[None]

        if len(poses.shape) == len(trans.shape):
            poses = rearrange(poses, "b l (p t) -> b l p t", t=3)

        matrix_poses = to_matrix(input_pose_rep, poses)

        save_shape_bs_len = matrix_poses.shape[:-3]
        nposes = reduce(operator.mul, save_shape_bs_len, 1)

        if matrix_poses.shape[-3] == 52:
            nohands = False
        elif matrix_poses.shape[-3] == 22:
            nohands = True
        else:
            raise NotImplementedError("Could not parse the poses.")

        # Reshaping
        matrix_poses = matrix_poses.reshape((nposes, *matrix_poses.shape[-3:]))
        global_orient = matrix_poses[:, 0]

        if trans is None:
            trans = torch.zeros(
                (*save_shape_bs_len, 3), dtype=poses.dtype, device=poses.device
            )

        trans_all = trans.reshape((nposes, *trans.shape[-1:]))

        body_pose = matrix_poses[:, 1:22]
        left_hand_pose = None
        right_hand_pose = None
        if self.model_type == "smplh":
            if nohands:
                # The training data used by STMC removes hand pose parameters.
                left_hand_pose = self.smplh.left_hand_mean.reshape(15, 3)
                left_hand_pose = to_matrix("axisangle", left_hand_pose)
                left_hand_pose = left_hand_pose[None].repeat((nposes, 1, 1, 1))

                right_hand_pose = self.smplh.right_hand_mean.reshape(15, 3)
                right_hand_pose = to_matrix("axisangle", right_hand_pose)
                right_hand_pose = right_hand_pose[None].repeat((nposes, 1, 1, 1))
            else:
                hand_pose = matrix_poses[:, 22:]
                left_hand_pose = hand_pose[:, :15]
                right_hand_pose = hand_pose[:, 15:]

        n = len(body_pose)

        if betas is not None:
            if len(betas.shape) == 1:
                # repeat betas
                betas = repeat(betas, "x -> b x", b=len(global_orient))
            else:
                # need to implement
                __import__("ipdb").set_trace()

        parameters = {
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": trans_all,
            "betas": betas,
        }
        if self.model_type == "smplh":
            parameters["left_hand_pose"] = left_hand_pose
            parameters["right_hand_pose"] = right_hand_pose

        # run smplh model, split by chunks to fit in memory
        outputs = []
        for smpl_output in call_by_chunks(self.smplh, n, batch_size, parameters):
            outputs.append(extract_data(smpl_output, jointstype, self.model_type))

        if jointstype != "both":
            outputs = torch.cat(outputs)
            outputs = outputs.reshape((*save_shape_bs_len, *outputs.shape[1:]))

            if needs_to_squeeze:
                outputs = outputs.squeeze(0)
            return outputs
        else:
            out = []
            for idx in range(2):
                output = torch.cat([x[idx] for x in outputs])
                output = output.reshape((*save_shape_bs_len, *output.shape[1:]))

                if needs_to_squeeze:
                    output = output.squeeze(0)
                out.append(output)
            return (*out,)


def find_body_model_path(path: str) -> tuple[str, str]:
    candidates = []

    env_path = os.environ.get("STMC_BODY_MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    candidates.append(path)

    this_file = os.path.abspath(__file__)
    stmc_root = os.path.dirname(os.path.dirname(os.path.dirname(this_file)))
    repo_root = os.path.dirname(stmc_root)
    candidates.extend(
        [
            os.path.join(stmc_root, "deps", "smplh"),
            os.path.join(
                repo_root,
                "LHM_3dnav",
                "pretrained_models",
                "human_model_files",
                "smplh",
            ),
            os.path.join(
                repo_root,
                "LHM_3dnav",
                "pretrained_models",
                "human_model_files",
                "smplx",
            ),
            os.path.join(
                repo_root,
                "LHM_3dnav",
                "pretrained_models",
                "human_model_files",
                "smpl",
            ),
        ]
    )

    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)
        model_type = detect_body_model_type(candidate)
        if model_type is not None:
            if model_type == "smpl":
                warnings.warn(
                    "Falling back to SMPL body models because no SMPL-H files were found. "
                    "This is compatible with STMC's body-only motion representation, but "
                    "hand articulation will not be available.",
                    stacklevel=2,
                )
            return candidate, model_type

    checked_paths = ", ".join(checked) if checked else path
    raise FileNotFoundError(
        "Could not find a compatible body model folder. Checked: "
        f"{checked_paths}. Set STMC_BODY_MODEL_PATH to a folder containing "
        "SMPLH_*.npz, SMPLX_*.npz, or SMPL_*.pkl files."
    )


def detect_body_model_type(path: str) -> Optional[str]:
    if not os.path.isdir(path):
        return None

    smplh_files = [
        f"SMPLH_{gender}.{ext}"
        for gender in ("NEUTRAL", "MALE", "FEMALE")
        for ext in ("npz", "pkl")
    ]
    if any(os.path.exists(os.path.join(path, filename)) for filename in smplh_files):
        return "smplh"

    smplx_files = [f"SMPLX_{gender}.npz" for gender in ("NEUTRAL", "MALE", "FEMALE")]
    if any(os.path.exists(os.path.join(path, filename)) for filename in smplx_files):
        return "smplx"

    smpl_files = [f"SMPL_{gender}.pkl" for gender in ("NEUTRAL", "MALE", "FEMALE")]
    if any(os.path.exists(os.path.join(path, filename)) for filename in smpl_files):
        return "smpl"

    return None


def load_smplh_gender(
    gender, smplh_folder, jointstype, batch_size, device, input_pose_rep="axisangle"
):
    if gender != "gendered":
        # only load one
        smplh = SMPLH(
            smplh_folder,
            input_pose_rep=input_pose_rep,
            jointstype=jointstype,
            gender=gender,
            batch_size=batch_size,
        ).to(device)
        return smplh

    # else load all of them
    smplh = {
        g: SMPLH(
            smplh_folder,
            input_pose_rep=input_pose_rep,
            jointstype=jointstype,
            gender=g,
            batch_size=batch_size,
        ).to(device)
        for g in ["male", "female", "neutral"]
    }
    return smplh
