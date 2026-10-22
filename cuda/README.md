# CUDA Hardware Validation

This folder contains the CUDA programs used for the real-GPU validation experiments in the project.

## Files

### `phase_mixed_gpu_test.cu`

Implements the phase-changing shared-memory bank-conflict experiment. It compares a conflict-free reference workload with a workload containing changing stride patterns.

The experiment uses:

- 256 blocks
- 32 threads per block
- 100 phase repetitions
- 100 iterations per phase
- 10 warm-up runs
- 50 measured runs
- Stride sequence: `1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1`

This experiment was executed on NVIDIA RTX 5070 and Tesla T4 GPUs to measure the runtime effect of shared-memory bank conflicts.

### `fair_layout_test.cu`

Implements the matrix-transpose hardware experiment comparing an unpadded shared-memory layout with a padded layout.

The experiment uses:

- Matrix size: `4096 × 4096`
- Baseline shared-memory tile: `32 × 32`
- Padded shared-memory tile: `32 × 33`
- 10 warm-up runs
- 50 measured runs
- Alternating execution order between baseline and padded kernels
- Output comparison for correctness

The padding changes the shared-memory access layout to reduce bank conflicts while preserving the computation result.

## Scope

These CUDA experiments validate that shared-memory bank conflicts can have a measurable effect on execution time on real GPU hardware.

They do not implement the proposed Normal/XOR-A/XOR-B adaptive bank-mapping controller on physical NVIDIA hardware. CUDA does not provide direct control over the GPU's physical shared-memory bank-mapping function.

The adaptive mapping controller itself is evaluated separately using the project's Python simulation.
