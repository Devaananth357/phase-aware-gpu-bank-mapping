# Phase-Aware GPU Bank Mapping

A research project investigating **phase-aware adaptive bank mapping for reducing GPU shared-memory bank conflicts**.

The project explores whether the mapping between shared-memory addresses and memory banks can adapt when memory-access patterns change during execution, instead of relying on one fixed mapping for the complete workload.

The final design combines phase-aware mapping selection with **safe data-lifetime boundaries, deferred switching, and boundary revalidation** to prevent unsafe remapping while shared-memory data is still active.

---

## Research Paper

**A Phase-Aware Adaptive Bank Mapping Approach for Reducing GPU Shared Memory Bank Conflicts**

The complete research paper is available in:

```text
paper/conference-Paper.pdf
```

---

## Motivation

GPU shared memory is divided into multiple banks that can operate in parallel.

When threads in a warp access addresses distributed across different banks, the accesses can be handled efficiently. When several threads access different addresses that map to the same bank, a **bank conflict** can occur and the accesses may need to be serialized.

Different address patterns can respond differently to different bank mappings.

A mapping that works well for one access pattern may not remain suitable when the memory-access pattern changes during execution.

This project studies whether the mapping can adapt to these changing phases.

---

## Main Contribution

The main contribution of this work is a **phase-aware adaptive bank-mapping controller** that can change the selected mapping when the memory-access pattern changes during execution.

The final controller also considers the lifetime of shared-memory data.

Mapping changes are allowed only at safe data-lifetime boundaries. If a mapping change is requested while data is still active, the request is deferred.

At the next safe boundary, the controller does not blindly apply the old request. The mapping decision is evaluated again and is applied only if it is still useful.

This prevents stale mapping decisions from being applied after the access pattern has already changed.

---

## Repository Structure

```text
phase-aware-gpu-bank-mapping/
│
├── README.md
├── LICENSE
├── realism-validation.py
│
├── cuda/
│   ├── README.md
│   ├── phase_mixed_gpu_test.cu
│   └── fair_layout_test.cu
│
├── results/
│   └── final_results.txt
│
└── paper/
    └── conference-Paper.pdf
```

---

# Simulation Model

The adaptive controller is evaluated using a Python-based shared-memory conflict simulator.

The final model uses:

```text
Shared-memory banks : 32
Threads per warp    : 32
```

Three experimental bank mappings are compared:

- Normal
- XOR-A
- XOR-B

The XOR mappings are simplified research mappings used to study controller behavior. They are **not intended to reproduce the exact internal bank-selection logic of NVIDIA GPUs**.

---

## Mapping Functions

### Normal Mapping

```text
bank = address mod 32
```

Python implementation:

```python
def normal_mapping(address):
    return address % NUM_BANKS
```

### XOR-A Mapping

```text
low5  = address & 31
high5 = (address >> 5) & 31

bank = low5 XOR high5
```

Python implementation:

```python
def xor_a_mapping(address):
    low5 = address & 0b11111
    high5 = (address >> 5) & 0b11111
    return low5 ^ high5
```

### XOR-B Mapping

```text
low5 = address & 31
mid5 = (address >> 3) & 31

bank = low5 XOR mid5
```

Python implementation:

```python
def xor_b_mapping(address):
    low5 = address & 0b11111
    mid5 = (address >> 3) & 0b11111
    return low5 ^ mid5
```

---

# Conflict-Cost Metric

For each simulated warp memory instruction, the addresses of the active threads are mapped to the 32 banks.

The **conflict degree** is the largest number of active accesses assigned to the same bank.

For example:

```text
Bank 0 -> 1 access
Bank 1 -> 1 access
Bank 2 -> 4 accesses
Bank 3 -> 2 accesses
```

The conflict degree is:

```text
4
```

The total conflict cost is the sum of these conflict degrees across the workload.

```text
Total Conflict Cost =
sum of conflict degrees across memory instructions
```

A lower value represents fewer simulated bank conflicts.

**Conflict cost is an experimental metric. It is not GPU execution time, clock cycles, or energy consumption.**

---

# Adaptive Controller

The final controller was developed through **35 iterative experiments**.

The development process explored:

- history-based mapping selection
- future lookahead
- stability voting
- switching thresholds
- minimum hold time
- causal watchdog monitoring
- emergency switching
- recovery locking
- safe data-lifetime boundaries
- deferred switching
- boundary revalidation
- correctness checking

The final evaluation compares three approaches.

### Best Static Mapping

One mapping is used for the complete workload.

The best of Normal, XOR-A, and XOR-B is used as the static baseline.

### Unrestricted Adaptive Controller

The controller can change mappings according to changing access behavior without requiring every switch to occur at a safe data-lifetime boundary.

It is used as an adaptive reference.

### Lifetime-Safe Adaptive Controller

This is the final correctness-oriented controller.

It adds:

- safe data-lifetime boundaries
- blocked-switch deferral
- pending mapping storage
- boundary revalidation
- stale-request rejection
- safe-switch correctness checking

---

# Safe Runtime Remapping

Changing an address-to-bank mapping while previously written shared-memory data is still active can create a correctness problem.

A later access to the same logical address could be interpreted using a different mapping.

The final controller therefore applies mapping changes only at **safe data-lifetime boundaries**.

Conceptually:

```text
Shared-memory Region A
        |
        | data is active
        |
        v
Region A ends
        |
        | SAFE BOUNDARY
        v
Mapping may change
        |
        v
Shared-memory Region B
```

In the current simulation, these safe boundaries are supplied by the experimental workload model.

Automatic detection of shared-memory data lifetime is outside the scope of the current implementation.

---

## Deferred Decisions and Boundary Revalidation

A useful mapping change may be requested while the controller is not at a safe boundary.

Instead of performing an unsafe switch, the controller stores the request.

When the next safe boundary is reached, the controller evaluates the current mapping situation again.

```text
Mapping change requested
        |
        v
Safe boundary available?
      /   \
    Yes    No
     |      |
     |      v
     |   Defer request
     |      |
     |      v
     |   Wait for boundary
     |      |
     +------+
        |
        v
Revalidate decision
        |
    +---+---+
    |       |
Useful    Stale
    |       |
 Apply    Reject
```

This is important because the access pattern may change while the controller is waiting for a safe remapping opportunity.

---

# Validation Workloads

Four final workloads are used by `realism-validation.py`.

## Matrix Transpose

A kernel-derived 32 × 32 tiled matrix-transpose shared-memory access pattern.

The model includes row-write and column-read phases.

## Parallel Reduction

A kernel-derived tree-style reduction pattern with decreasing active-thread counts.

## Prefix Scan

A kernel-derived Blelloch prefix-scan pattern containing upsweep and downsweep phases.

## Phase-Mixed Workload

A synthetic phase-changing stress workload constructed from shared-memory stride patterns.

The phase sequence includes:

```text
3, 32, 24, 5, 16, 24
```

Different phases can favor different mappings.

The Phase-Mixed workload is intentionally designed to test whether the controller can respond when the preferred mapping changes during execution.

It is a **synthetic validation workload** and is not claimed to represent one specific CUDA kernel.

---

# Simulation Results

The final conflict-cost results are:

| Benchmark | Best Static | Unrestricted Adaptive | Lifetime-Safe Adaptive |
|---|---:|---:|---:|
| Matrix Transpose | 128 | 128 | 128 |
| Parallel Reduction | 768 | 768 | 768 |
| Prefix Scan | 1536 | 1536 | 1536 |
| Phase-Mixed | 3296 | 2353 | 1559 |

For Matrix Transpose, Parallel Reduction, and Prefix Scan, the adaptive controller matches the best static mapping.

The strongest adaptive benefit appears on the Phase-Mixed workload, where the preferred mapping changes across phases.

The lifetime-safe controller reduces conflict cost from:

```text
3296 -> 1559
```

giving:

```text
(3296 - 1559) / 3296 × 100 = 52.70%
```

Therefore, the final reported simulation result is:

**52.70% reduction in simulated conflict cost compared with the best static mapping on the Phase-Mixed workload.**

This value is **not a 52.70% GPU execution-time speedup**.

---

# Safe-Switch Correctness

The simulator records the mapping used for every memory instruction.

A mapping change is considered unsafe if it occurs outside a safe data-lifetime boundary.

For the final lifetime-safe controller:

```text
Unsafe applied mapping switches = 0
```

This verifies that the final controller does not apply mapping changes while the experimental model marks previous shared-memory data as active.

---

# Real GPU Validation

The adaptive Normal/XOR-A/XOR-B controller cannot be directly installed on the tested NVIDIA GPUs through CUDA because CUDA does not expose direct control over the physical shared-memory bank-mapping function.

Real CUDA experiments were therefore performed separately to verify that shared-memory bank conflicts have a measurable effect on execution time.

The experiments were executed on:

- NVIDIA GeForce RTX 5070
- NVIDIA Tesla T4

These experiments validate the **performance relevance of bank conflicts**, not the hardware performance of the proposed adaptive controller itself.

---

## Experiment A — Phase-Changing Bank-Conflict Test

A CUDA workload containing changing shared-memory stride patterns was compared with a conflict-free reference workload.

The stride sequence was:

```text
1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1
```

Configuration:

```text
Blocks             : 256
Threads per block  : 32
Phase repetitions  : 100
Iterations/phase   : 100
Warm-up runs       : 10
Measured runs      : 50
```

### Results

| GPU | Conflict-Free Mean | Phase-Changing Mean | Slowdown |
|---|---:|---:|---:|
| Tesla T4 | 3.0667 ms | 15.5467 ms | 5.0695× |
| RTX 5070 | 1.8169 ms | 4.4465 ms | 2.4473× |

The phase-changing workload was substantially slower than the conflict-free reference on both GPUs.

Source:

```text
cuda/phase_mixed_gpu_test.cu
```

---

## Experiment B — Matrix-Transpose Padding Test

A second hardware experiment used a:

```text
4096 × 4096
```

matrix transpose.

The baseline implementation used:

```text
shared-memory tile = 32 × 32
```

while the padded implementation used:

```text
shared-memory tile = 32 × 33
```

Adding one column changes the shared-memory access layout and reduces bank conflicts during transposed accesses.

The experiment used:

```text
Warm-up runs  : 10
Measured runs : 50
```

The baseline and padded execution order was alternated between measured runs to reduce systematic order bias.

### Results

| GPU | Baseline | Padded | Runtime Reduction | Speedup |
|---|---:|---:|---:|---:|
| RTX 5070 | 0.4481 ms | 0.3299 ms | 26.38% | 1.3584× |
| Tesla T4 | 1.4176 ms | 0.6695 ms | 52.78% | 2.1175× |

The output of the baseline and padded implementations was also compared for correctness.

```text
Output mismatches = 0
```

Source:

```text
cuda/fair_layout_test.cu
```

The similarity between the **52.70% simulation conflict-cost reduction** and the **52.78% Tesla T4 matrix-transpose runtime reduction** should not be interpreted as a direct quantitative correlation. They are results from different experiments measuring different quantities.

---

# Running the Simulation

## Requirements

Python 3 is required.

No external Python packages are required for the final simulation.

Check your Python installation:

```bash
python --version
```

Run:

```bash
python realism-validation.py
```

or:

```bash
python3 realism-validation.py
```

The script reports:

- static mapping costs
- best static mapping
- unrestricted adaptive results
- lifetime-safe adaptive results
- mapping switches
- blocked mapping requests
- revalidation statistics
- stale requests
- safe-boundary information
- unsafe switch count
- percentage improvement

---

# Running the CUDA Experiments

The CUDA experiments require:

- NVIDIA CUDA-capable GPU
- CUDA Toolkit
- `nvcc`

Compile the phase-changing test:

```bash
nvcc cuda/phase_mixed_gpu_test.cu -O3 -o phase_mixed_gpu_test
```

Run:

```bash
./phase_mixed_gpu_test
```

Compile the matrix-transpose test:

```bash
nvcc cuda/fair_layout_test.cu -O3 -o fair_layout_test
```

Run:

```bash
./fair_layout_test
```

On Windows, the generated programs can be run as:

```text
phase_mixed_gpu_test.exe
fair_layout_test.exe
```

Exact timings may differ between GPUs, CUDA versions, driver versions, clock states, and system configurations.

---

# Interpretation of the Results

The project contains two separate forms of evaluation.

### Simulation

The simulator evaluates whether adaptive selection between Normal, XOR-A, and XOR-B can reduce the experimental **conflict-cost metric**.

The main result is:

```text
52.70% conflict-cost reduction
```

on the synthetic Phase-Mixed workload.

### Real GPU Testing

The CUDA experiments evaluate whether reducing shared-memory bank conflicts can improve actual kernel execution time.

The matrix-transpose experiment measured runtime reductions of:

```text
RTX 5070 : 26.38%
Tesla T4 : 52.78%
```

These measurements demonstrate the performance relevance of shared-memory bank conflicts but do not represent measured speedups of the proposed adaptive mapping controller.

---

# Limitations

The current work has several limitations.

### Simplified Mapping Functions

Normal, XOR-A, and XOR-B are experimental mapping models and do not reproduce the exact internal mapping functions of commercial NVIDIA GPUs.

### Conflict-Cost Simulation

The adaptive-controller evaluation uses conflict cost rather than a cycle-accurate GPU performance model.

### Synthetic Phase-Mixed Workload

The strongest adaptive result occurs on the synthetic Phase-Mixed workload. The 52.70% result should therefore not be interpreted as a universal improvement across GPU applications.

### Safe-Boundary Knowledge

Safe data-lifetime boundaries are supplied by the experimental model.

A real implementation would require hardware tracking, compiler support, programmer annotations, or another mechanism for determining when previous shared-memory data is no longer active.

### Lookahead Availability

The final deterministic validation assumes future addresses are available to the simulator.

A practical hardware implementation would require prediction, compiler information, queued memory requests, or another mechanism for obtaining useful future-access information.

### No Physical Adaptive-Mapping Implementation

The proposed adaptive mapping controller has not been integrated into physical GPU hardware.

The CUDA experiments validate the runtime effect of shared-memory bank conflicts separately.

---

# Future Work

Possible extensions include:

- RTL implementation using Verilog or SystemVerilog
- integration with GPGPU-Sim or another GPU architecture simulator
- automatic safe-boundary detection
- realistic future-access prediction
- hardware area, timing, and power analysis
- controller-overhead analysis
- additional CUDA workloads
- larger benchmark suites
- compiler-assisted mapping decisions
- architectural implementation of runtime-selectable bank mappings

---

# Key Result

The central result of the project is:

```text
Phase-Mixed workload

Best Static Mapping       : 3296
Unrestricted Adaptive     : 2353
Lifetime-Safe Adaptive    : 1559

Conflict-Cost Reduction   : 52.70%
Unsafe Applied Switches   : 0
```

The result demonstrates that phase-aware adaptive mapping can reduce simulated bank-conflict cost when the preferred mapping changes during execution, while the lifetime-safe controller prevents mapping changes outside the supplied safe data-lifetime boundaries.

---

# References

The project was developed with reference to previous research on GPU shared-memory bank conflicts and adaptive scratchpad-memory mapping, including:

1. C. Gou and G. N. Gaydadjiev, **“Addressing GPU On-Chip Shared Memory Bank Conflicts Using Elastic Pipeline.”**

2. G.-J. van den Braak, J. Gómez-Luna, J. M. González-Linares, H. Corporaal, and N. Guil, **“Configurable XOR Hash Functions for Banked Scratchpad Memories in GPUs.”**

3. F. Han, L. Li, K. Wang, F. Feng, H. Pan, J. Sha, and J. Lin, **“An Access Pattern Based Adaptive Mapping Function for GPGPU Scratchpad Memory.”**

The work by Han et al. is particularly related because it also investigates adaptive scratchpad-memory mapping. This project focuses on **phase changes during execution, lifetime-safe remapping, deferred decisions, and boundary revalidation**.

---

## Author

**Devaananth P**  
B.Tech Computer Science and Engineering  
SRM Institute of Science and Technology
