# ComfyUI market image candidate

The image builds official ComfyUI `v0.32.0` from revision
`c2bcbecd82ec5ae66594340b395c24ef0217b238`, with PyTorch 2.7.1
CUDA 12.8 wheels for the target RTX 5090. It includes the upstream GPL-3.0
license and a resolved Python package list. No model or custom node is bundled.
The CUDA 12.8.1 cuDNN Ubuntu 24.04 base uses a fixed `linux/amd64` digest.

Release version `0.1.0` is a VerdantFlare delivery version, separate from the
upstream `0.32.0`. The release workflow checks that its Registry tag is unused,
then builds and pushes for `linux/amd64`. Record the CI image digest in the
Hub Chart review before any target installation. Local checks cannot prove GPU
operators, license clearance of every dependency, or Station ingress.

Source and Kubernetes policy decisions live in the design repository under
`docs/design/center/market/examples/comfyui/`.
