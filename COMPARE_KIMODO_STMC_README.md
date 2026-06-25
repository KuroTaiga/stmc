# Kimodo vs STMC Comparison Runbook

This runbook documents the local scripts around `compare_kimodo_stmc.py`.
Run commands from the repository root:

```bash
cd /home/lenovo/TeleNav_RenderPipe
```

## What the main script does

`external/stmc/compare_kimodo_stmc.py` takes a text prompt file in:

```text
prompt text # duration_seconds
```

It then:

1. Generates matching Kimodo motions.
2. Generates matching STMC motions.
3. Renders Kimodo and STMC skeleton videos.
4. Renders Kimodo and STMC mesh videos.
5. Builds labeled side-by-side comparison videos.
6. Builds optional overlay comparison videos unless `--skip_overlay` is used.
7. Exports STMC `*_smpl.npz` into per-frame LHM++ JSON in each STMC output folder.

Default environments are `kimodo` and `stmc`. Override them with
`--kimodo_env` and `--stmc_env` if your conda env names differ.

## Required setup

Before running comparisons, make sure these are available:

- Conda env `kimodo` with Kimodo dependencies.
- Conda env `stmc` with STMC dependencies.
- `ffmpeg` on `PATH`.
- STMC pretrained run directory:
  `external/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d`
- SMPL-X neutral asset. The helper scripts search these locations:
  `external/kimodo/kimodo/assets/skeletons/smplx22/SMPLX_NEUTRAL.npz`,
  `pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz`,
  `external/LHM_3dnav/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz`,
  and `external/LHM_pp/pretrained_models/Damo_XR_Lab/LHMPP-Prior/human_model_files/smplx/SMPLX_NEUTRAL.npz`.

## Prompt files

Useful local prompt files:

- `external/stmc/eval_prompts/kimodo_stmc_hand_actions_short.txt` - short hand-action smoke test.
- `external/stmc/eval_prompts/kimodo_stmc_hand_actions.txt` - fuller hand-action eval set.
- `external/stmc/eval_prompts/single_actions_text.txt` - general single-action eval prompts.
- `external/stmc/eval_prompts/single_actions_text_medium.txt` - medium single-action set.
- `external/stmc/eval_prompts/basic_actions.txt` - minimal sanity prompt.
- `external/stmc/eval_prompts/single_actions_timeline.txt` - STMC timeline format, not the main comparison script's text-file format.

## Main comparison commands

### Dry-run before generating

Use this to check selected prompt indices and output paths without generation:

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/kimodo_stmc_hand_actions_short.txt \
  --indices 0 1 \
  --out_dir outputs/kimodo_stmc_compare/hand_actions_short \
  --dry_run
```

### Short eval prompt run

Use this for the fastest end-to-end check:

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/kimodo_stmc_hand_actions_short.txt \
  --out_dir outputs/kimodo_stmc_compare/hand_actions_short
```

### Full hand-action eval prompts

Use this when evaluating the hand-action prompt set:

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/kimodo_stmc_hand_actions.txt \
  --out_dir outputs/kimodo_stmc_compare/hand_actions
```

### General single-action eval prompts

Run all general prompts:

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/single_actions_text.txt \
  --out_dir outputs/kimodo_stmc_compare/single_actions
```

Run only selected 0-based prompt indices:

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/single_actions_text.txt \
  --indices 0 4 6 \
  --out_dir outputs/kimodo_stmc_compare/single_actions_selected
```

### Fast comparison without overlay videos

Overlay rendering is slower. Use this when side-by-side skeleton and mesh videos are enough:

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/kimodo_stmc_hand_actions.txt \
  --out_dir outputs/kimodo_stmc_compare/hand_actions_no_overlay \
  --skip_overlay
```

### Regenerate existing outputs

The script reuses existing outputs by default. Add `--overwrite` when you want
to delete and regenerate selected prompt folders:

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/kimodo_stmc_hand_actions_short.txt \
  --out_dir outputs/kimodo_stmc_compare/hand_actions_short \
  --overwrite
```

### Use explicit envs, seed, checkpoint, and guidance

```bash
python external/stmc/compare_kimodo_stmc.py \
  --text_file external/stmc/eval_prompts/kimodo_stmc_hand_actions_short.txt \
  --out_dir outputs/kimodo_stmc_compare/hand_actions_short_seed1234 \
  --kimodo_env kimodo \
  --stmc_env stmc \
  --seed 1234 \
  --kimodo_model kimodo-smplx-rp \
  --kimodo_diffusion_steps 100 \
  --kimodo_transition_frames 5 \
  --stmc_ckpt last \
  --stmc_guidance 2.5 \
  --stmc_device cuda
```

## Output layout

For `--out_dir outputs/kimodo_stmc_compare/hand_actions_short`, outputs look like:

```text
outputs/kimodo_stmc_compare/hand_actions_short/
  prompts/
    000_a_person_is_waving_with_the_right_hand.txt
    000_a_person_is_waving_with_the_right_hand.json
    kimodo_batch_manifest.json
    stmc_batch_prompts.txt
  kimodo/
    000_a_person_is_waving_with_the_right_hand/
      frames/*.json
      amass.npz
      kimodo_motion.npz
      kimodo_skeleton.mp4
      kimodo_mesh.mp4
      meta.json
  stmc/
    000_a_person_is_waving_with_the_right_hand/
      frames/*.json
      stmc_joints.npy
      stmc_verts.npy
      stmc_smpl.npz
      stmc_skeleton.mp4
      stmc_mesh.mp4
  comparisons/
    000_a_person_is_waving_with_the_right_hand_skeleton_compare.mp4
    000_a_person_is_waving_with_the_right_hand_mesh_compare.mp4
    000_a_person_is_waving_with_the_right_hand_overlay_compare.mp4
```

## Helper scripts by situation

### Kimodo only: direct prompt to frames, AMASS, NPZ, and videos

```bash
conda run -n kimodo python external/stmc/kimodo_smplx_to_json.py \
  --out_dir outputs/kimodo_only/wave_right_hand \
  --prompts "a person is waving with the right hand" \
  --durations 3.0 \
  --seed 1234 \
  --save_amass \
  --save_motion_npz \
  --z_up
```

### Kimodo only: one eval prompt by index

```bash
conda run -n kimodo python external/stmc/kimodo_smplx_to_json.py \
  --out_dir outputs/kimodo_only/single_action_004_wave \
  --text_file external/stmc/eval_prompts/single_actions_text.txt \
  --text_indices 4 \
  --seed 1234 \
  --save_amass \
  --save_motion_npz \
  --z_up
```

### Kimodo only: render videos from an existing `kimodo_motion.npz`

```bash
conda run -n kimodo python external/stmc/kimodo_smplx_to_json.py \
  --out_dir outputs/kimodo_only/single_action_004_wave \
  --motion_npz outputs/kimodo_only/single_action_004_wave/kimodo_motion.npz \
  --video_name kimodo_skeleton.mp4 \
  --mesh_video_name kimodo_mesh.mp4
```

### STMC only: generate motions from a text prompt file

```bash
conda run -n stmc python external/stmc/generate.py \
  run_dir=external/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d \
  timeline=external/stmc/eval_prompts/kimodo_stmc_hand_actions_short.txt \
  input_type=text \
  value_from=smpl \
  fast=false \
  seed=1234 \
  ckpt=last \
  guidance=2.5 \
  device=cuda \
  render_joints=false \
  render_smpl=false
```

Generated arrays are written under:

```text
external/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/generations/<prompt_stem>_last_text_to_motion/
```

Typical files are:

```text
<prompt_stem>_text_0.npy
<prompt_stem>_text_0_verts.npy
<prompt_stem>_text_0_smpl.npz
```

### STMC only: render skeleton and mesh videos

Render the joints array:

```bash
conda run -n stmc python external/stmc/render.py \
  path=external/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/generations/kimodo_stmc_hand_actions_short_last_text_to_motion/kimodo_stmc_hand_actions_short_text_0.npy \
  out_path=outputs/stmc_only/hand_action_000/stmc_skeleton.mp4 \
  fps=20.0
```

Render the mesh vertices array:

```bash
conda run -n stmc python external/stmc/render.py \
  path=external/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/generations/kimodo_stmc_hand_actions_short_last_text_to_motion/kimodo_stmc_hand_actions_short_text_0_verts.npy \
  out_path=outputs/stmc_only/hand_action_000/stmc_mesh.mp4 \
  fps=20.0
```

### STMC only: convert `*_smpl.npz` into LHM++ frame JSON

For one file:

```bash
conda run -n stmc python external/stmc/stmc_npz_to_lhmpp_json_seq.py \
  --input external/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/generations/kimodo_stmc_hand_actions_short_last_text_to_motion/kimodo_stmc_hand_actions_short_text_0_smpl.npz \
  --out_dir outputs/stmc_only/hand_action_000/frames
```

For a directory of STMC `.npz` files:

```bash
conda run -n stmc python external/stmc/stmc_npz_to_lhmpp_json_seq.py \
  --input_dir external/stmc/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d/generations/kimodo_stmc_hand_actions_short_last_text_to_motion \
  --out_dir outputs/stmc_only/converted_frames
```

### Existing outputs: build only an overlay comparison video

Use this if Kimodo and STMC assets already exist and you only need the overlay:

```bash
conda run -n kimodo python external/stmc/render_overlay_compare.py \
  --kimodo_motion_npz outputs/kimodo_stmc_compare/hand_actions_short/kimodo/000_a_person_is_waving_with_the_right_hand/kimodo_motion.npz \
  --stmc_joints_npy outputs/kimodo_stmc_compare/hand_actions_short/stmc/000_a_person_is_waving_with_the_right_hand/stmc_joints.npy \
  --stmc_verts_npy outputs/kimodo_stmc_compare/hand_actions_short/stmc/000_a_person_is_waving_with_the_right_hand/stmc_verts.npy \
  --out_path outputs/kimodo_stmc_compare/hand_actions_short/comparisons/000_a_person_is_waving_with_the_right_hand_overlay_compare.mp4 \
  --prompt_text "a person is waving with the right hand" \
  --duration_s 3.0 \
  --fps 30 \
  --figsize 6.0
```

### Interactive Gradio comparison app

Run the app from an environment that has Gradio and the Kimodo-side preview
dependencies. The app itself calls the configured Kimodo and STMC conda envs
when generating:

```bash
conda run -n kimodo python external/stmc/compare_gradio_app.py \
  --host 127.0.0.1 \
  --port 7860 \
  --kimodo_env kimodo \
  --stmc_env stmc
```

Add `--share` only when you need a public Gradio share URL.

## Common fixes

- `ModuleNotFoundError: einops` or other Kimodo imports: run Kimodo helper
  commands through `conda run -n kimodo ...`.
- `ModuleNotFoundError: gradio`: install Gradio in the env used to launch
  `compare_gradio_app.py`, or launch from an env that already has it.
- Existing videos are being reused: add `--overwrite`.
- Overlay generation is taking too long: add `--skip_overlay`.
- Missing `SMPLX_NEUTRAL.npz`: place the asset in one of the paths listed in
  Required setup.
- Missing STMC generation arrays: check that `--stmc_run_dir` points at the
  pretrained run directory and that the `stmc` env can run `generate.py`.
