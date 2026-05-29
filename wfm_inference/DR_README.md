# DiffusionRenderer Inverse — Lilypad Entrypoint

Runs Cosmos **DiffusionRenderer Inverse** (RGB video → albedo/normal/depth/
roughness/metallic G-buffers) single-GPU on Lilypad generic workloads. This is a
**separate image** from the cosmos_transfer2 multiview path (see `README.md`):
DR is cosmos_predict1 lineage with an incompatible dependency tree, so it gets its
own clean image built from `docker/diffusion_renderer.Dockerfile`.

The workload config lives at
`adp/services/wfm/lilypad_workload_configs/cosmos_diffusion_renderer_inference.yaml`
in the applied3 repo.

## Architecture

Lilypad generic workload → Ray cluster:
- **Head node** (CPU) runs `wfm_inference.dr_lilypad_entrypoint.run`.
- **GPU worker** (1× A100) runs `_run_dr_on_gpu`: downloads weights from OCI once,
  then runs the requested mode (no torchrun — DR inverse is single-process),
  uploads outputs + a `succeed.txt` marker.

Two modes (`config["mode"]`, default `smoke`):
- **smoke** (phase 1): `wfm_inference.dr_smoke_inference` on a synthetic zeros video.
- **segment** (phase 2): downloads real exported frames from OCI, then
  `wfm_inference.run_dr_on_segment`, a thin wrapper that drives the **upstream**
  `cosmos_predict1.diffusion.inference.inference_inverse_renderer` in `--group_mode
  folder` (one camera subfolder = one clip → all cameras in a single pipeline load).
  We do not re-implement frame loading / resize / chunking / gamma — upstream does it.
  Long clips are handled by upstream `--overlap_n_frames` / `--chunk_mode` (plain
  overlap-chunking, **not** BrickDiffusion). Albedo is saved raw (display-ready);
  inverse-gamma for linear `A*S+R` compositing is a phase-3 reconstruction concern.
  Outputs are curated into `<camera>/albedo/NNNN.basecolor.jpg` and side-by-side
  `<camera>/mosaic/NNNN.png` (`[input | albedo]`).

DR needs only **1 GPU (~27 GB)**. There is no HF-cache step: zero T5 embeddings
are passed at inference, so no text encoder is required; only the two weight dirs
are needed.

## Weights (staged in OCI)

| Model | HF repo | Size | Gated |
|-------|---------|------|-------|
| Inverse DiT | `nvidia/Diffusion_Renderer_Inverse_Cosmos_7B` | ~27 GB | No |
| Video tokenizer | `nvidia/Cosmos-Tokenize1-CV8x8x8-720p` | ~1.9 GB | Yes |

Stage both under `s3://sensor-sim-wfm/checkpoints/diffusion_renderer/` so the
layout is:

```
checkpoints/diffusion_renderer/
  Diffusion_Renderer_Inverse_Cosmos_7B/...
  Cosmos-Tokenize1-CV8x8x8-720p/...
```

One-time copy from the existing staging bucket (different tenancy):

```bash
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
# pull (oci.phx creds)
aws s3 sync s3://onroad-perception-datasets/adp_neural_sim/diffusion_renderer/ /tmp/dr_weights/
# push (idskhu5vqvtl creds)
aws s3 sync /tmp/dr_weights/ s3://sensor-sim-wfm/checkpoints/diffusion_renderer/ \
  --endpoint-url https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com
```

## Build and push

Weights are NOT baked into the image (downloaded at runtime), so it stays lean.

```bash
docker login us-phoenix-1.ocir.io -u idskhu5vqvtl/caleb.levy@applied.co  # OCI auth token as pw
DOCKER_BUILDKIT=1 docker build -f docker/diffusion_renderer.Dockerfile \
  -t us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_dr_inverse_v0.0.1 .
docker push us-phoenix-1.ocir.io/idskhu5vqvtl/lilypad/sds:cosmos_dr_inverse_v0.0.1
```

Only the flat `lilypad/sds` repo is pullable by Lilypad (subdirs aren't in IAM),
so keep the `cosmos_dr_inverse_*` tag on that path. Update `docker_image` in the
applied3 YAML to match.

## Local iteration (RTX 5090 / sm_120)

```bash
DOCKER_BUILDKIT=1 docker build -f docker/diffusion_renderer.Dockerfile -t cosmos_dr_inverse:dev .
docker run --gpus all --ipc=host -v /tmp/dr_weights:/weights cosmos_dr_inverse:dev \
  python -m wfm_inference.dr_smoke_inference --checkpoint-dir /weights --output-dir /tmp/out
```

**Expected locally:** imports + weight-load succeed, then the denoise step hits
OOM or "no kernel image available for sm_120" (TE/megatron ship sm_80/90 kernels
only). Both are caught and printed as "image OK, run on A100". Any *Python*
import/config error is a real bug — fix it locally before going to the cluster.

## Launch on Lilypad

```bash
export AWS_ACCESS_KEY_ID=<oci-access-key>
export AWS_SECRET_ACCESS_KEY=<oci-secret-key>
lilypad workload launch \
  adp/services/wfm/lilypad_workload_configs/cosmos_diffusion_renderer_inference.yaml \
  --name caleb-dr-smoke-$(date +%s)
```

Use standard `AWS_*` vars (pass-through from the submitting shell; confirmed in
the Neural Sim Lilypad Cookbook). Verify success: `succeed.txt` and
`albedo_frame0.npy` appear under `s3://sensor-sim-wfm/inferences/dr_smoke_test/`,
with a peak-VRAM line (~27 GB) in `lilypad workload logs <id>`.

## Segment mode (phase 2 — real frames)

The phase-2 image tag is `cosmos_dr_inverse_v0.0.2` (adds `run_dr_on_segment.py` +
segment-mode entrypoint; rebuild + push when the wrapper changes).

1. Export frames from an AppliedGS dataset (applied3 side, local/CPU) and upload to OCI:
   ```bash
   bazel run //third_party/applied_gs/tools/appliedgs/experimental/intrinsics:export_segment_frames -- \
     --dataset-dir /applied2/tmp/pandaset_139 \
     --cameras front_camera,left_camera,back_camera \
     --out /applied2/tmp/dr_segments/pandaset_139
   aws s3 sync /applied2/tmp/dr_segments/pandaset_139 \
     s3://sensor-sim-wfm/inputs/dr_segments/pandaset_139/ \
     --profile oci.phx --endpoint-url $EP
   ```
2. Launch the segment workload (config sets `mode: segment`, input/output prefixes,
   `resize`, `chunk_mode`):
   ```bash
   lilypad workload launch \
     adp/services/wfm/lilypad_workload_configs/cosmos_diffusion_renderer_segment_inference.yaml \
     --name caleb-dr-segment-$(date +%s)
   ```
3. Download + scroll the mosaics:
   ```bash
   aws s3 sync s3://sensor-sim-wfm/inferences/dr_segments/pandaset_139/ \
     ~/dr_out/pandaset_139/ --profile oci.phx --endpoint-url $EP
   eog ~/dr_out/pandaset_139/front_camera/mosaic/
   ```

Local sm_120 dry-run of the wrapper exits **nonzero at the denoise step** (upstream
prints "no kernel image available for sm_120") — that's the expected local wall; it
confirms arg plumbing + output curation up to the GPU op. Run on A100 for real albedo.
