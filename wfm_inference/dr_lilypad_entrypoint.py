"""Lilypad entrypoint for Cosmos DiffusionRenderer (Inverse).

Mirrors wfm_inference/lilypad_entrypoint.py (Ray head -> GPU worker, OCI boto
download/upload) but runs cosmos_predict1 DiffusionRenderer instead of
cosmos_transfer2 multiview. DR inverse is single-GPU, single-process: the worker
claims num_gpus=1 and runs directly (no torchrun, no HF cache remap — DR needs only
its two OCI-staged weight dirs).

Two modes (config["mode"], default "smoke"):

  smoke   -- run dr_smoke_inference on a synthetic zeros video (phase 1 validation).
  segment -- download real exported frames from OCI, run run_dr_on_segment (drives the
             upstream folder-mode inverse renderer over all camera subfolders), and
             upload the curated albedo/ + mosaic/ tree (phase 2).

Common config keys:
    weights_bucket / weights_prefix    OCI bucket + prefix with the two DR model subdirs.
    output_bucket / output_prefix      OCI destination for outputs.

smoke-only:  num_steps (15), num_frames (57).
segment-only:
    input_bucket / input_prefix        OCI prefix with <camera>/NNNN.png subfolders.
    num_steps (15), num_frames (57), overlap_n_frames (8), chunk_mode ("first"),
    passes (["basecolor"]), resize ([H, W] or omit), save_video ("True").
"""
import logging
import os

import ray

logger = logging.getLogger(__name__)

# Persistent dir on the worker so the ~29 GB weights survive across jobs.
_WORKER_CACHE_DIR = "/tmp/dr_worker_cache"


def _oci_client():
    """boto3 S3 client configured for OCI S3-compat (payload signing, no v4 checksum)."""
    import boto3
    import botocore.config

    oci_config = botocore.config.Config(
        s3={"payload_signing_enabled": True},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=oci_config,
    )


def _download_prefix(client, bucket: str, prefix: str, dest) -> int:
    """Download every object under s3://bucket/prefix/ into dest, preserving layout."""
    from pathlib import Path

    prefix = prefix.rstrip("/")
    dest = Path(dest)
    paginator = client.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue
            local = dest / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local))
            n += 1
    return n


def _upload_dir(client, local_dir, bucket: str, prefix: str) -> int:
    """Upload every file under local_dir to s3://bucket/prefix/, preserving layout."""
    from pathlib import Path

    local_dir = Path(local_dir)
    prefix = prefix.rstrip("/")
    files = [p for p in sorted(local_dir.rglob("*")) if p.is_file()]
    for path in files:
        key = f"{prefix}/{path.relative_to(local_dir)}".lstrip("/")
        client.upload_file(str(path), bucket, key)
    return len(files)


def _ensure_weights(client, config: dict, log) -> "object":
    """Download both DR model subdirs once into the persistent cache; return the dir."""
    from pathlib import Path

    weights_bucket = config["weights_bucket"]
    weights_prefix = config["weights_prefix"].rstrip("/")
    checkpoint_dir = Path(_WORKER_CACHE_DIR) / "weights"
    if (checkpoint_dir / "Diffusion_Renderer_Inverse_Cosmos_7B").exists():
        log.info("DR weights already cached at %s", checkpoint_dir)
    else:
        log.info("Downloading DR weights s3://%s/%s -> %s", weights_bucket, weights_prefix, checkpoint_dir)
        n = _download_prefix(client, weights_bucket, weights_prefix, checkpoint_dir)
        log.info("Downloaded %d weight file(s)", n)
    return checkpoint_dir


@ray.remote(num_gpus=1)
def _run_dr_on_gpu(config: dict) -> None:
    """Runs on the GPU worker: download weights once, then run the requested mode."""
    import logging
    import subprocess
    import tempfile
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    client = _oci_client()
    checkpoint_dir = _ensure_weights(client, config, log)
    mode = config.get("mode", "smoke")
    output_bucket = config["output_bucket"]
    output_prefix = config["output_prefix"].rstrip("/")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "outputs"
        output_dir.mkdir(parents=True)

        if mode == "smoke":
            cmd = [
                "python", "-m", "wfm_inference.dr_smoke_inference",
                "--checkpoint-dir", str(checkpoint_dir),
                "--output-dir", str(output_dir),
                "--num-steps", str(config.get("num_steps", 15)),
                "--num-frames", str(config.get("num_frames", 57)),
            ]
        elif mode == "segment":
            input_dir = Path(tmpdir) / "inputs"
            input_dir.mkdir(parents=True)
            n_in = _download_prefix(client, config["input_bucket"], config["input_prefix"], input_dir)
            log.info("Downloaded %d input frame file(s) -> %s", n_in, input_dir)
            cmd = [
                "python", "-m", "wfm_inference.run_dr_on_segment",
                "--checkpoint-dir", str(checkpoint_dir),
                "--input-root", str(input_dir),
                "--output-dir", str(output_dir),
                "--passes", *config.get("passes", ["basecolor"]),
                "--num-frames", str(config.get("num_frames", 57)),
                "--overlap-n-frames", str(config.get("overlap_n_frames", 8)),
                "--chunk-mode", str(config.get("chunk_mode", "first")),
                "--num-steps", str(config.get("num_steps", 15)),
                "--save-video", str(config.get("save_video", "True")),
            ]
            resize = config.get("resize")
            if resize:
                cmd += ["--resize-resolution", str(resize[0]), str(resize[1])]
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected 'smoke' or 'segment'.")

        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            raise RuntimeError(f"{cmd[2]} exited with code {result.returncode}")

        n_out = _upload_dir(client, output_dir, output_bucket, output_prefix)
        log.info("Uploaded %d output file(s) to s3://%s/%s", n_out, output_bucket, output_prefix)
        client.put_object(Body=b"", Bucket=output_bucket, Key=f"{output_prefix}/succeed.txt")
        log.info("Upload complete.")


def run(config: dict) -> None:
    """Lilypad entrypoint: dispatch DR inference to a single GPU worker."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ray.init(address="auto")
    logger.info("Dispatching DR inverse (%s mode) to GPU worker (num_gpus=1)", config.get("mode", "smoke"))
    ray.get(_run_dr_on_gpu.remote(config))
    logger.info("DR inference complete.")
