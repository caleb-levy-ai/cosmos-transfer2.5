"""Run Cosmos DiffusionRenderer (Inverse) albedo on real segment frames (phase 2).

Thin wrapper around the upstream real-frame inference script
``cosmos_predict1.diffusion.inference.inference_inverse_renderer``. We do NOT
re-implement frame loading, resize/crop, chunking, or the diffusion pipeline — the
upstream script already provides all of that:

  * ``--group_mode folder`` groups one input subfolder = one clip, so a single run
    over ``<input_root>/<camera>/*.png`` processes every camera with one pipeline load.
  * ``--resize_resolution H W`` then center-crop to ``--height/--width`` (704x1280).
  * ``--overlap_n_frames`` / ``--chunk_mode`` handle clips longer than the 57-frame
    context window (plain overlap-chunking; NOT BrickDiffusion).
  * per-frame G-buffer JPGs at
    ``<save>/gbuffer_frames/<camera>/<chunk>.<idx:04d>.<pass>.jpg``.

This wrapper invokes that script, then reorganizes the outputs into
``<output_dir>/<camera>/albedo/NNNN.basecolor.jpg`` and builds side-by-side
``[input | albedo]`` mosaics at ``<output_dir>/<camera>/mosaic/NNNN.png`` so the
downloaded folder is immediately scrollable in an image viewer.

Albedo is saved raw (display-ready uint8 as the model emits it). Inverse-gamma for
linear ``A*S+R`` compositing is a phase-3 reconstruction concern, not applied here.

Usage (on an A100; locally on sm_120 it reaches the denoise wall, which is expected):
    python -m wfm_inference.run_dr_on_segment \
        --checkpoint-dir /weights --input-root /inputs/pandaset_139 \
        --output-dir /out/pandaset_139 --resize-resolution 720 1280 --chunk-mode first
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
from PIL import Image

# DiffusionRenderer source clone location (configs load relative to repo root).
_DR_REPO_ROOT = "/cosmos-dr"
_DR_MODEL_NAME = "Diffusion_Renderer_Inverse_Cosmos_7B"
_INFERENCE_MODULE = "cosmos_predict1.diffusion.inference.inference_inverse_renderer"
# Upstream writes per-frame jpgs under this subdir of --video_save_folder.
_GBUFFER_SUBDIR = "gbuffer_frames"


def _parse_jpg_global_index(filename: str, stride: int) -> int | None:
    """Map an upstream jpg name '<chunk>.<idx>.<pass>.jpg' to a global frame index.

    global = chunk * stride + idx, where stride = num_frames - overlap. Returns None
    if the name does not match the expected pattern.
    """
    parts = filename.split(".")
    if len(parts) < 4 or parts[-1].lower() != "jpg":
        return None
    try:
        chunk, idx = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return chunk * stride + idx


def run_upstream_inference(args: argparse.Namespace, save_folder: str) -> None:
    """Invoke the upstream inverse-renderer over all camera subfolders at once."""
    # Pipeline loads its config from paths relative to the repo root.
    os.chdir(_DR_REPO_ROOT)
    cmd = [
        "python", "-m", _INFERENCE_MODULE,
        "--checkpoint_dir", args.checkpoint_dir,
        "--diffusion_transformer_dir", _DR_MODEL_NAME,
        "--dataset_path", args.input_root,
        "--group_mode", "folder",
        "--inference_passes", *args.passes,
        "--num_video_frames", str(args.num_frames),
        "--overlap_n_frames", str(args.overlap_n_frames),
        "--chunk_mode", args.chunk_mode,
        "--height", str(args.height),
        "--width", str(args.width),
        "--num_steps", str(args.num_steps),
        "--save_image=True",
        f"--save_video={args.save_video}",
        "--offload_diffusion_transformer",
        "--offload_tokenizer",
        "--offload_text_encoder_model",
        "--offload_guardrail_models",
        "--video_save_folder", save_folder,
    ]
    if args.resize_resolution is not None:
        cmd += ["--resize_resolution", str(args.resize_resolution[0]), str(args.resize_resolution[1])]
    print("Running upstream inverse renderer:\n  " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"{_INFERENCE_MODULE} exited with code {result.returncode}")


def _blend_chunk_frames(contributions: list[tuple[int, str]], num_frames: int) -> np.ndarray:
    """Triangular-weighted blend of a frame predicted by multiple overlapping chunks.

    A frame's weight within its chunk peaks at the chunk center and falls to ~0 at the
    chunk edges (w = min(idx+1, num_frames-idx)). Where two chunks overlap, this
    cross-fades them — the leaving chunk fades out as the entering chunk fades in — so
    there is no hard seam at the boundary. (Mitigation pending Brick Diffusion, which
    would instead share information across chunks during denoising.)
    """
    arrays, weights = [], []
    for idx, path in contributions:
        arrays.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))
        weights.append(float(min(idx + 1, num_frames - idx)))
    stack = np.stack(arrays, axis=0)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1, 1)
    blended = (stack * w).sum(axis=0) / w.sum()
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def build_outputs(args: argparse.Namespace, save_folder: str) -> None:
    """Reorganize upstream jpgs into albedo/ (overlap-blended) + [input|albedo] mosaics.

    Also writes <output_dir>/manifest.json recording the resize/crop transform so
    phase-3 reconstruction can align each albedo frame back to the original RGB grid.
    """
    gbuffer_root = os.path.join(save_folder, _GBUFFER_SUBDIR)
    if not os.path.isdir(gbuffer_root):
        raise FileNotFoundError(
            f"No {_GBUFFER_SUBDIR}/ produced under {save_folder}; upstream wrote nothing."
        )
    stride = max(1, args.num_frames - args.overlap_n_frames)
    manifest: dict = {
        "passes": args.passes,
        "num_frames": args.num_frames,
        "overlap_n_frames": args.overlap_n_frames,
        "chunk_mode": args.chunk_mode,
        "crop_height": args.height,
        "crop_width": args.width,
        "resize_resolution": args.resize_resolution,
        "cameras": {},
    }
    for camera in sorted(os.listdir(gbuffer_root)):
        camera_gbuffer = os.path.join(gbuffer_root, camera)
        if not os.path.isdir(camera_gbuffer):
            continue
        albedo_dir = os.path.join(args.output_dir, camera, "albedo")
        mosaic_dir = os.path.join(args.output_dir, camera, "mosaic")
        os.makedirs(albedo_dir, exist_ok=True)
        os.makedirs(mosaic_dir, exist_ok=True)

        # Group every chunk's prediction by global frame index; overlap frames get >1.
        contributions: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for jpg in (f for f in os.listdir(camera_gbuffer) if f.endswith(".basecolor.jpg")):
            global_index = _parse_jpg_global_index(jpg, stride)
            if global_index is None:
                continue
            idx_in_chunk = int(jpg.split(".")[1])
            contributions[global_index].append((idx_in_chunk, os.path.join(camera_gbuffer, jpg)))

        # The last chunk is padded up to num_frames, so it can emit "frames" past the
        # real input count (duplicates of the final frame). Drop those phantom frames
        # so staged albedo matches the real RGB frames exactly.
        input_camera_dir = os.path.join(args.input_root, camera)
        n_input = len([f for f in os.listdir(input_camera_dir) if f.endswith(".png")]) \
            if os.path.isdir(input_camera_dir) else 0
        if n_input:
            contributions = {g: c for g, c in contributions.items() if g < n_input}

        n_blended = 0
        for global_index, contribs in sorted(contributions.items()):
            if len(contribs) == 1:
                albedo = np.asarray(Image.open(contribs[0][1]).convert("RGB"), dtype=np.uint8)
            else:
                albedo = _blend_chunk_frames(contribs, args.num_frames)
                n_blended += 1
            albedo_img = Image.fromarray(albedo, mode="RGB")
            albedo_img.save(os.path.join(albedo_dir, f"{global_index:04d}.basecolor.jpg"), quality=95)
            input_png = os.path.join(args.input_root, camera, f"{global_index:04d}.png")
            if os.path.exists(input_png):
                _write_mosaic(input_png, albedo_img, os.path.join(mosaic_dir, f"{global_index:04d}.png"))

        native_hw = _native_resolution(os.path.join(args.input_root, camera))
        manifest["cameras"][camera] = {
            "num_albedo_frames": len(contributions),
            "num_overlap_blended": n_blended,
            "native_height": native_hw[0],
            "native_width": native_hw[1],
            "albedo_to_original": _alignment_transform(native_hw, args),
        }
        print(f"  {camera}: {len(contributions)} albedo frames ({n_blended} overlap-blended)", flush=True)

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def _native_resolution(input_camera_dir: str) -> tuple[int | None, int | None]:
    """(height, width) of the first input PNG in a camera dir, or (None, None)."""
    pngs = sorted(f for f in os.listdir(input_camera_dir) if f.endswith(".png")) \
        if os.path.isdir(input_camera_dir) else []
    if not pngs:
        return None, None
    with Image.open(os.path.join(input_camera_dir, pngs[0])) as im:
        return im.height, im.width


def _alignment_transform(native_hw: tuple[int | None, int | None], args: argparse.Namespace) -> dict | None:
    """Map an albedo pixel (704x1280) back to the original RGB grid for phase 3.

    Pipeline is: original (native) --resize--> resize_resolution --center-crop--> (H,W).
    So original = (albedo + crop_offset) * (native / resize). Returns the offsets/scales.
    """
    native_h, native_w = native_hw
    if native_h is None or args.resize_resolution is None:
        return None
    resize_h, resize_w = args.resize_resolution
    return {
        "crop_offset_y": (resize_h - args.height) // 2,
        "crop_offset_x": (resize_w - args.width) // 2,
        "scale_y": native_h / resize_h,
        "scale_x": native_w / resize_w,
    }


def _write_mosaic(input_png: str, albedo_img: Image.Image, out_png: str) -> None:
    """Write a side-by-side [input | albedo] PNG, both scaled to a common height."""
    inp = Image.open(input_png).convert("RGB")
    height = albedo_img.height
    if inp.height != height:
        inp = inp.resize((round(inp.width * height / inp.height), height), Image.BILINEAR)
    canvas = Image.new("RGB", (inp.width + albedo_img.width, height))
    canvas.paste(inp, (0, 0))
    canvas.paste(albedo_img, (inp.width, 0))
    canvas.save(out_png)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiffusionRenderer albedo on real segment frames.")
    parser.add_argument("--checkpoint-dir", required=True, help="Dir with DR DiT + tokenizer subdirs.")
    parser.add_argument("--input-root", required=True, help="Dir with <camera>/NNNN.png subfolders.")
    parser.add_argument("--output-dir", required=True, help="Where albedo/ + mosaic/ are written.")
    parser.add_argument("--passes", nargs="+", default=["basecolor"], help="G-buffer passes to infer.")
    parser.add_argument("--num-frames", type=int, default=57, help="Context window (frames per chunk).")
    parser.add_argument("--overlap-n-frames", type=int, default=8, help="Overlap between chunks.")
    parser.add_argument("--chunk-mode", default="first", choices=["first", "all", "drop_last"])
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--resize-resolution", type=int, nargs=2, default=None,
                        metavar=("H", "W"), help="Resize before center-crop, e.g. 720 1280.")
    parser.add_argument("--save-video", default="True", choices=["True", "False"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = os.path.join(args.checkpoint_dir, _DR_MODEL_NAME)
    if not os.path.isdir(weights_path):
        print(f"ERROR: DR weights not found at {weights_path}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # Run upstream into a temp save dir, then curate into output_dir. On sm_120 the
    # upstream subprocess prints "no kernel image available" and exits nonzero at the
    # denoise step (the expected local-validation wall); on A100/H100 it completes.
    with tempfile.TemporaryDirectory() as save_folder:
        run_upstream_inference(args, save_folder)
        build_outputs(args, save_folder)
    print("DR segment inference DONE.")


if __name__ == "__main__":
    main()
