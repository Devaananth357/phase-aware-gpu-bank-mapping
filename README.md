# Phase-Aware GPU Bank Mapping

A research project on **adaptive GPU shared-memory bank mapping** with phase-aware switching, safe runtime remapping, and conflict-cost evaluation.

The project studies whether a GPU shared-memory bank mapping can adapt during execution instead of using one fixed mapping for the complete workload.

The final design was reached after **35 iterative experiments**, beginning with simple history-based mapping selection and gradually moving toward future-aware control, causal watchdogs, emergency switching, safe lifetime boundaries, deferred decisions, boundary revalidation, and correctness checking.

---

## Project Motivation

GPU shared memory is divided into multiple memory banks that can operate in parallel.

When threads in a warp access different addresses that map to different banks, the accesses can be handled efficiently. However, when several threads access addresses that fall into the same bank, a **bank conflict** occurs.

These conflicts may force memory requests to be handled sequentially and can reduce the efficiency of shared-memory operations.

A simple example is:

```text
Thread 0 -> Bank 0
Thread 1 -> Bank 1
Thread 2 -> Bank 2
Thread 3 -> Bank 3
```

These accesses can happen in parallel.

But if the mapping becomes:

```text
Thread 0 -> Bank 0
Thread 1 -> Bank 0
Thread 2 -> Bank 0
Thread 3 -> Bank 0
```

the accesses conflict.

Different address mappings can reduce these conflicts for different access patterns.

The main problem is that a mapping which works well during one phase of execution may not remain the best mapping during another phase.

This project studies whether the bank mapping can change dynamically according to the current memory-access pattern.

---

# Main Idea

Instead of using a single fixed bank mapping for the entire workload, the controller chooses between three mappings:

- Normal Mapping
- XOR-A Mapping
- XOR-B Mapping

The controller evaluates the access pattern and attempts to select the mapping that produces the lowest conflict cost.

The final controller combines:

- future lookahead
- short and long stability voting
- switching thresholds
- minimum hold time
- causal watchdog monitoring
- emergency switching
- recovery locking
- safe lifetime boundaries
- deferred mapping requests
- boundary revalidation
- correctness checking

---

# Shared-Memory Model

The final experiments use:

```text
Shared-memory banks : 32
Threads per warp    : 32
```

This matches the 32-thread warp model used throughout the final evaluation.

---

# Mapping Functions

Three bank-mapping functions are evaluated.

## Normal Mapping

The normal mapping assigns an address directly using modulo bank selection.

```text
bank = address mod 32
```

In the implementation:

```python
def normal_mapping(address):
    return address % NUM_BANKS
```

---

## XOR-A Mapping

XOR-A combines the lower five address bits with the next group of higher address bits.

```text
low5  = address[4:0]
high5 = address[9:5]

bank = low5 XOR high5
```

Implementation:

```python
def xor_a_mapping(address):
    low5 = address & 0b11111
    high5 = (address >> 5) & 0b11111
    return low5 ^ high5
```

---

## XOR-B Mapping

XOR-B uses another address-bit combination.

```text
low5 = address[4:0]
mid5 = selected middle bits

bank = low5 XOR mid5
```

Implementation:

```python
def xor_b_mapping(address):
    low5 = address & 0b11111
    mid5 = (address >> 3) & 0b11111
    return low5 ^ mid5
```

These XOR mappings are simplified research models used for experimentation and are not intended to represent the exact bank-selection logic of a commercial GPU.

---

# Conflict-Cost Model

The main metric used in the project is **conflict degree**.

For one warp memory instruction, the simulator checks how many active threads map to each bank.

The conflict degree is the highest number of accesses assigned to one bank.

For example:

```text
Bank 0 -> 1 thread
Bank 1 -> 1 thread
Bank 2 -> 4 threads
Bank 3 -> 2 threads
```

The conflict degree is:

```text
4
```

The total conflict cost of a workload is calculated by adding the conflict degree of every memory instruction.

```text
Total Conflict Cost =
sum of conflict degree for all memory instructions
```

A smaller value means fewer simulated bank conflicts.

This project measures **conflict cost**, not GPU execution time.

---

# Development Process

The final design was not created in one step.

A total of **35 experiments** were performed while developing the controller.

The experiments gradually revealed several important problems and led to changes in the controller design.

---

## Early Experiments

The first experiments focused on selecting a mapping using recent memory-access history.

The controller examined a sliding window and attempted to determine which mapping performed best.

Different approaches were tested, including:

- history-based mapping selection
- stability detection
- fallback mappings
- switching penalties
- cooldown periods
- random-like workload detection

These approaches worked on some workloads but often reacted too slowly when the access pattern changed.

This revealed a major limitation:

> Past accesses alone may not provide enough information when the memory pattern changes quickly.

---

# Future-Aware Mapping

The original direction of the project was then changed from history-only prediction to **future lookahead**.

Instead of only asking:

```text
What mapping performed best recently?
```

the controller also considers:

```text
What mapping is expected to work best for the upcoming accesses?
```

Future windows of different sizes were evaluated.

The final controller mainly uses:

```text
5-access lookahead
10-access lookahead
```

These windows provide information about which mapping may be better in the near future.

---

# Hybrid Mapping Selection

A hybrid selector was developed to prevent the controller from switching too aggressively.

The controller maintains the winning mapping from multiple lookahead windows.

Two voting levels are used.

## High-Stability Mode

The controller checks the most recent ten decisions.

A mapping must receive at least:

```text
9 out of 10 votes
```

before being considered highly stable.

---

## Medium-Stability Mode

If high stability is not available, the controller checks the last five decisions.

A mapping must receive:

```text
3 out of 5 votes
```

to become the candidate.

---

## Baseline Mode

If neither voting condition is satisfied, the winner from the current short lookahead window is used as the candidate.

---

# Switching Threshold

A mapping is not changed just because another mapping is slightly better.

The candidate must provide a minimum improvement.

The final threshold is:

```text
15%
```

This reduces unnecessary mapping changes.

---

# Minimum Hold Time

After changing mappings, the controller must remain in the selected mapping for a minimum amount of time before another normal switch is allowed.

The final value is:

```text
MIN_HOLD = 2
```

This helps reduce rapid switching between mappings.

---

# Causal Watchdog

Future information may not always be enough, especially when prediction becomes unavailable or the current mapping starts performing badly.

A **causal watchdog** was introduced.

The watchdog only examines already completed memory accesses.

The final watchdog window is:

```text
16 memory instructions
```

If the current mapping performs significantly worse than another mapping over this previous window, the watchdog can trigger a mapping change.

The watchdog threshold used in the final controller is:

```text
30%
```

The watchdog is causal because it only uses past completed accesses.

---

# Recovery Lock

After the watchdog changes the mapping, a recovery period is activated.

The final recovery lock is:

```text
8 instructions
```

Normal hybrid switching is restricted during this period.

The goal is to prevent the controller from immediately reversing a watchdog decision.

---

# Emergency Switching

Testing showed that the recovery lock could sometimes prevent the controller from reacting to a sudden severe change in the memory pattern.

To solve this, an **emergency switching path** was added.

If the short future window shows that another mapping provides a sufficiently large improvement, the controller may switch even while the recovery lock is active.

The emergency threshold is:

```text
30%
```

This became the final **Unrestricted Adaptive Controller**.

---

# Why Safe Remapping Became Necessary

At this stage, another problem became important.

Changing the bank mapping while shared-memory data is still active can change the physical location associated with an address.

That can cause previously written data to be read using a different mapping.

A simple bank-level mapping switch therefore cannot automatically preserve correctness.

This means runtime remapping cannot safely happen at arbitrary instructions.

---

# Safe Data-Lifetime Boundaries

The final solution allows mapping changes only at **safe lifetime boundaries**.

A safe boundary represents a point where the previous shared-memory region is no longer needed.

At that point, the mapping may safely change before the next region begins.

Conceptually:

```text
Region A
    |
    | Shared-memory data active
    |
Region A ends
    |
    | SAFE REMAPPING POINT
    |
Region B begins
```

This prevents mappings from changing while previous shared-memory data is still live.

---

# Deferred Switching

A mapping request may occur when the controller is not currently at a safe boundary.

Instead of applying the switch immediately, the request can be deferred.

The controller stores:

```text
requested mapping
request source
```

The request may come from:

- watchdog
- emergency path
- normal hybrid selector

---

# Why Blind Deferred Switching Failed

One development experiment attempted to simply save a blocked request and apply it at the next safe point.

This performed poorly.

The reason was simple:

> The access pattern may have changed before the safe boundary is reached.

A mapping decision that was correct earlier may become stale.

This led to the final improvement.

---

# Boundary Revalidation

Before applying a deferred mapping request, the controller checks whether the decision is still valid.

This process is called **boundary revalidation**.

At a safe boundary:

```text
Deferred request
        |
        v
Re-evaluate current memory behavior
        |
        +---- still useful ----> Apply mapping
        |
        +---- stale -----------> Reject request
```

Different request types are revalidated differently.

### Watchdog Request

The watchdog is revalidated using recent completed accesses.

### Emergency Request

The emergency request is rechecked using the current short future window.

### Normal Hybrid Request

The normal request is evaluated again using the hybrid voting and lookahead mechanism.

This prevents old decisions from being blindly applied.

---

# Final Controllers

The final validation compares three approaches.

## Best Static Mapping

The best mapping is selected after evaluating the complete workload.

It represents the strongest fixed-mapping baseline.

---

## Unrestricted Adaptive Controller

This controller can dynamically change mappings using:

- lookahead
- hybrid voting
- causal watchdog
- recovery lock
- emergency switching

It does not restrict every mapping change to data-lifetime boundaries.

This controller represents the maximum adaptive behavior used as a reference.

---

## Lifetime-Safe Adaptive Controller

This is the final correctness-oriented controller.

It adds:

- safe lifetime boundaries
- blocked-switch deferral
- pending mapping storage
- boundary revalidation
- stale-request rejection
- safe-switch correctness checking

This is the main final design of the project.

---

# Final Validation

The consolidated validation script is:

```text
realism_validation.py
```

It contains all final mappings, controllers, benchmarks, conflict-cost functions, safe-boundary logic, and result reporting.

No external Python packages are required.

---

# Validation Workloads

Four final workloads are included.

---

## 1. Tiled Matrix Transpose

A 32 × 32 unpadded shared-memory tile is modeled.

Two important memory phases are generated:

```text
Row write
Column read
```

The previous tile is considered dead when the next tile begins.

This provides natural safe-remapping boundaries between tiles.

---

## 2. Parallel Reduction

A standard contiguous tree-style reduction access pattern is modeled.

Active threads reduce as the stride becomes smaller.

Inactive lanes are represented separately so that they do not create artificial conflicts.

Safe remapping is allowed only between completed blocks.

This benchmark acts as a useful control workload because the normal reduction access pattern is largely conflict free.

---

## 3. Blelloch Prefix Scan

The benchmark models the shared-memory access structure of a Blelloch prefix scan.

Both phases are included:

```text
Upsweep
Downsweep
```

The shared-memory state remains active throughout one scan block.

Therefore remapping is allowed only between completed blocks.

---

## 4. Phase-Mixed Shared-Memory Stress Benchmark

The Phase-Mixed benchmark was created specifically to evaluate phase-changing memory behavior.

It combines realistic stride-based shared-memory access motifs.

Example phase strides include:

```text
3
32
24
5
16
24
```

Different strides favor different bank mappings.

For example:

```text
Stride 3  -> Normal can be favorable
Stride 32 -> XOR-A can be favorable
Stride 24 -> XOR-B can be favorable
```

Each region owns fresh shared-memory state.

Therefore the beginning of a new region acts as a safe remapping boundary.

This benchmark is synthetic and is not claimed to represent one specific CUDA kernel.

---

# Final Results

The final conflict-cost results are:

| Benchmark | Best Static | Unrestricted Adaptive | Lifetime-Safe Adaptive |
|---|---:|---:|---:|
| Matrix Transpose | 128 | 128 | 128 |
| Parallel Reduction | 768 | 768 | 768 |
| Prefix Scan | 1536 | 1536 | 1536 |
| Phase-Mixed | 3296 | 2353 | 1559 |

---

# Matrix Transpose Result

```text
Best Static            : 128
Unrestricted Adaptive  : 128
Lifetime-Safe Adaptive : 128
```

The adaptive controllers match the best static mapping.

No unnecessary improvement is claimed because one mapping is already suitable for the entire workload.

---

# Parallel Reduction Result

```text
Best Static            : 768
Unrestricted Adaptive  : 768
Lifetime-Safe Adaptive : 768
```

All mappings produce the same final conflict cost in the validation model.

The controller therefore avoids introducing a penalty.

---

# Prefix Scan Result

```text
Best Static            : 1536
Unrestricted Adaptive  : 1536
Lifetime-Safe Adaptive : 1536
```

The adaptive controllers again match the best static mapping.

---

# Phase-Mixed Result

```text
Best Static            : 3296
Unrestricted Adaptive  : 2353
Lifetime-Safe Adaptive : 1559
```

The lifetime-safe adaptive controller produces the lowest conflict cost.

The improvement compared with the best static mapping is:

```text
(3296 - 1559) / 3296 × 100
```

which gives:

```text
52.70%
```

Therefore:

> The lifetime-safe adaptive controller reduces simulated conflict cost by 52.70% compared with the best static mapping on the Phase-Mixed benchmark.

This value should not be interpreted as a 52.70% GPU execution-time speedup.

---

# Correctness Validation

Correctness was treated as a separate requirement instead of assuming that any mapping switch is safe.

The final validation records the mapping used at every memory instruction.

A switch is considered unsafe if:

```text
mapping[i] != mapping[i - 1]
```

and the current instruction is not marked as a safe lifetime boundary.

The final safe controller produced:

```text
Unsafe switches = 0
```

for the final validation workloads.

---

# Important Findings

The development process produced several important observations.

## 1. One Mapping Is Not Always Best

Different stride patterns may favor different address mappings.

This creates an opportunity for adaptive mapping.

---

## 2. History Alone Can React Too Slowly

Controllers using only previous accesses can struggle when memory behavior changes quickly.

Future lookahead improved responsiveness.

---

## 3. Switching Too Often Is Also Harmful

Aggressive switching creates unstable behavior.

Voting, thresholds, hold time, and recovery locking were therefore added.

---

## 4. Recovery Locks Need an Emergency Path

A fixed lock can prevent adaptation when a severe phase transition occurs.

Emergency bypass allows the controller to react to large predicted improvements.

---

## 5. Correctness Restricts Adaptation

A controller cannot safely change mappings anywhere it wants.

Shared-memory lifetime must be considered.

---

## 6. Deferred Decisions Can Become Stale

A mapping request that was useful when it was created may no longer be useful at the next safe boundary.

Boundary revalidation solves this problem by checking the request again.

---

## 7. Safe-Boundary Density Matters

Adaptive mapping is most useful when:

```text
memory access patterns change
AND
safe remapping opportunities exist
```

If safe boundaries are extremely far apart, the controller may lose much of its ability to react to short phases.

---

# Research Scope

The project should be understood as a **simulation and architecture-design study**.

The final experiments evaluate:

- shared-memory address traces
- bank mappings
- conflict degree
- adaptive mapping decisions
- safe remapping logic
- mapping-switch correctness

The project does not claim measured GPU execution-time improvement.

---

# Limitations

Several limitations remain.

### Simulated Conflict Cost

The main metric is conflict cost rather than actual GPU execution time.

---

### Simplified XOR Mapping

XOR-A and XOR-B are research mapping functions.

They are not exact models of the internal bank-mapping logic of NVIDIA or other commercial GPUs.

---

### Kernel-Derived Traces

Matrix transpose, parallel reduction, and prefix scan use algorithm-derived address traces.

They are not memory traces captured directly from physical GPU hardware.

---

### Synthetic Phase-Mixed Benchmark

The largest adaptive improvement appears on the Phase-Mixed workload.

This benchmark is synthetic, although it is built from realistic stride-based shared-memory access motifs.

---

### Safe-Boundary Knowledge

The simulator is told where shared-memory lifetime boundaries occur.

A real implementation would require compiler support, programmer annotations, hardware tracking, or another method for identifying safe remapping points.

---

### Lookahead Availability

The final realism validation assumes future addresses are available for the deterministic traces.

Real hardware would require the addresses to be predictable from known addressing logic or some form of prediction.

---

# Possible Future Work

This project can be extended in several directions.

## Hardware Implementation

The controller could be implemented as an RTL design using Verilog or SystemVerilog.

Possible hardware blocks include:

```text
Address Pattern Monitor
        |
        v
Lookahead / Prediction Logic
        |
        v
Hybrid Mapping Selector
        |
        +------> Causal Watchdog
        |
        +------> Emergency Decision Logic
        |
        v
Safe Boundary Gate
        |
        v
Pending Mapping Register
        |
        v
Boundary Revalidation
        |
        v
Mapping Selector
        |
        +--> Normal
        +--> XOR-A
        +--> XOR-B
```

---

## GPGPU-Sim Integration

A future version could integrate the controller into GPGPU-Sim and measure:

- simulated execution cycles
- IPC
- memory stalls
- bank-conflict stalls
- performance overhead

---

## Real CUDA Testing

CUDA kernels could be used to compare software-level memory layouts and gather real timing measurements.

---

## Automatic Safe-Boundary Detection

Future work could investigate ways to identify when shared-memory data becomes dead.

Possible approaches include:

- compiler analysis
- programmer annotations
- kernel barriers
- shared-memory allocation boundaries
- hardware lifetime tracking

---

## Hardware Cost Analysis

A hardware version could also be synthesized to measure:

- area
- power
- timing
- controller state
- mapping-selection overhead

This would connect the project more directly with VLSI and GPU microarchitecture research.

---

# Repository Structure

The clean public repository contains only the final implementation and supporting project material.

```text
phase-aware-gpu-bank-mapping/
│
├── README.md
├── realism_validation.py
├── LICENSE
├── results/
│   └── final_results.txt
│
└── paper/
    └── conference-Paper.pdf
```

Intermediate experiment files are intentionally not included in the public repository.

They were used during controller development but are not required to reproduce the final validation.

---

# Running the Project

## Requirements

Only Python 3 is required.

No third-party packages are needed.

Check Python:

```bash
python --version
```

Run the final validation:

```bash
python realism_validation.py
```

On some systems:

```bash
python3 realism_validation.py
```

---

# Expected Output

The script prints:

- static mapping costs
- best static mapping
- unrestricted adaptive cost
- lifetime-safe adaptive cost
- number of mapping switches
- percentage improvement against static mapping
- blocked mapping requests
- revalidation attempts
- applied revalidations
- rejected revalidations
- stale requests
- unsafe switch count
- final benchmark interpretation

---

# Example Final Result

The most important final result is the Phase-Mixed workload:

```text
Best Static            = 3296
Unrestricted Adaptive  = 2353
Lifetime-Safe Adaptive = 1559

Improvement vs Best Static = 52.70%
Unsafe Switches             = 0
```

---

# Research Paper

The repository also includes the research paper:

**A Phase-Aware Adaptive Bank Mapping Approach for Reducing GPU Shared Memory Bank Conflicts**

The paper explains:

- motivation
- related work
- proposed methodology
- experimental setup
- results
- limitations
- conclusion

---

# Related Research

The work was developed after studying research related to GPU shared-memory bank conflicts and adaptive bank mapping.

Important references include:

1. C. Gou and G. N. Gaydadjiev,  
   **“Addressing GPU On-Chip Shared Memory Bank Conflicts Using Elastic Pipeline.”**

2. G.-J. van den Braak, J. Gómez-Luna, J. M. González-Linares, H. Corporaal, and N. Guil,  
   **“Configurable XOR Hash Functions for Banked Scratchpad Memories in GPUs.”**

3. F. Han, L. Li, K. Wang, F. Feng, H. Pan, J. Sha, and J. Lin,  
   **“An Access Pattern Based Adaptive Mapping Function for GPGPU Scratchpad Memory.”**

4. A. Horga, A. Rezine, S. Chattopadhyay, P. Eles, and Z. Peng,  
   **“Symbolic Identification of Shared Memory Based Bank Conflicts for GPUs.”**

The work by Han et al. is particularly related because it also studies adaptive scratchpad-memory mapping.

The focus of this project is on repeated phase changes, safe runtime remapping, deferred decisions, and boundary revalidation.

---

# Important Reporting Note

The results in this repository should be described as:

> **simulated shared-memory conflict-cost results using synthetic and kernel-derived address traces**

They should not be described as:

```text
real NVIDIA GPU execution-time measurements
hardware-measured speedups
hardware-captured GPU traces
```

---

# Author

**Devaananth P**

Department of Computer Science and Engineering  
SRM Institute of Science and Technology  
Chennai, India

---

# License

This project is released under the **MIT License**.

See the `LICENSE` file for details.

---

# Project Status

```text
Research idea              : Completed
Controller development     : Completed
35 iterative experiments   : Completed
Safe-remapping design      : Completed
Correctness validation     : Completed
Kernel-derived validation  : Completed
Final result freeze        : Completed
Research paper             : Completed
Public repository cleanup  : Completed
```

The current repository represents the **final frozen version** of the research project.
