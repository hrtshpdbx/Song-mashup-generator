#!/usr/bin/env bash
# Build the All-In-One structure analyser as a local Cog image that runs on RTX 40-series GPUs,
# then check that it really uses the GPU.
#
#   scripts/build_allin1_image.sh            # builds and tests image "allin1-cu118"
#   scripts/build_allin1_image.sh my-tag     # custom tag (then: export MASHUP_ALLIN1_IMAGE=my-tag)
#
# Needs: docker (with the NVIDIA container toolkit), cog, git, ~15 GB free disk.
# The first build takes a while: NATTEN's CUDA kernels are compiled from source.
set -euo pipefail

TAG="${1:-allin1-cu118}"
UPSTREAM="https://github.com/sakemin/all-in-one.git"
COMMIT="bc7d092c4bb9231543e78e4742d7707597fd090a"   # branch `cog`, 2023-12-21
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ALLIN1_BUILD_DIR:-$REPO_ROOT/build/allin1-src}"

for tool in docker cog git; do
  command -v "$tool" >/dev/null || { echo "ERROR: '$tool' not found on PATH." >&2; exit 1; }
done

if [ ! -d "$BUILD_DIR/.git" ]; then
  echo "== Cloning $UPSTREAM into $BUILD_DIR"
  git clone --quiet "$UPSTREAM" "$BUILD_DIR"
fi
git -C "$BUILD_DIR" fetch --quiet origin cog
git -C "$BUILD_DIR" checkout --quiet --force "$COMMIT"
cp "$REPO_ROOT/third_party/allin1/cog.yaml" "$BUILD_DIR/cog.yaml"

echo "== Building image '$TAG' (cog.yaml from third_party/allin1/)"
( cd "$BUILD_DIR" && cog build -t "$TAG" )

echo "== Self-test inside the image (GPU required)"
docker run --rm -i --gpus all --entrypoint python "$TAG" - <<'PY'
import torch, numpy, madmom.ml.hmm  # noqa: F401
from natten.functional import natten1dqkrpb, natten1dav
assert torch.cuda.is_available(), "torch can't see the GPU inside the container"
print("numpy", numpy.__version__, "| torch", torch.__version__, "CUDA", torch.version.cuda,
      "|", torch.cuda.get_device_name(0))
spec = torch.stft(torch.randn(44100, device="cuda"), 2048, return_complex=True)
print("cuFFT ok", tuple(spec.shape))
q = torch.randn(1, 2, 64, 16, device="cuda")
rpb = torch.randn(2, 2 * 5 - 1, device="cuda")
attn = natten1dqkrpb(q, q, rpb, 5, 1).softmax(-1)
out = natten1dav(attn, q, 5, 1)
ref_attn = natten1dqkrpb(q.cpu(), q.cpu(), rpb.cpu(), 5, 1).softmax(-1)
err = (attn.cpu() - ref_attn).abs().max().item()
assert err < 1e-4, f"NATTEN GPU result differs from CPU by {err}"
print("NATTEN CUDA ok", tuple(out.shape), f"(max diff vs CPU {err:.1e})")
import os, glob
assert glob.glob("/root/.cache/torch/hub/checkpoints/955717e8-8726e21a.th"), "Demucs weights not baked in"
assert len(glob.glob("/root/.cache/huggingface/hub/models--taejunkim--allinone/snapshots/*/*.pth")) == 8, \
    "All-In-One checkpoints not baked in"
print("weights baked in: ok")
PY

echo
echo "Image '$TAG' is ready. The pipeline uses it by default if the tag is 'allin1-cu118';"
echo "otherwise run:  export MASHUP_ALLIN1_IMAGE=$TAG"
