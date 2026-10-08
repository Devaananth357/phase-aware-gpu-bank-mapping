# ============================================================
# FINAL REALISM VALIDATION
#
# Benchmarks:
#   1. Tiled Matrix Transpose
#   2. Parallel Reduction
#   3. Blelloch Prefix Scan
#   4. Phase-Mixed Shared-Memory Stress Benchmark
#
# Controllers:
#   Exp-29 : unrestricted adaptive controller
#   Exp-35 : lifetime-safe + deferred boundary revalidation
#
# NOTE:
# These are kernel-derived / algorithm-derived address traces.
# They are NOT hardware-captured NVIDIA traces.
# ============================================================

from collections import deque, Counter


# ============================================================
# GLOBAL CONFIG
# ============================================================

NUM_BANKS = 32
WARP_SIZE = 32
BANK_MASK = NUM_BANKS - 1

INITIAL_MAPPING = "XOR-A"

MIN_HOLD = 2
SWITCH_THRESHOLD = 0.15
EMERGENCY_THRESHOLD = 0.30

HIGH_STABILITY_VOTES = 9
MEDIUM_STABILITY_VOTES = 3

WATCHDOG_WINDOW = 16
WATCHDOG_THRESHOLD = 0.30
RECOVERY_LOCK = 8


# ============================================================
# BANK MAPPINGS
# ============================================================

def normal_mapping(address):
    return address % NUM_BANKS


def xor_a_mapping(address):

    low5 = address & BANK_MASK
    high5 = (address >> 5) & BANK_MASK

    return low5 ^ high5


def xor_b_mapping(address):

    low5 = address & BANK_MASK
    mid5 = (address >> 3) & BANK_MASK

    return low5 ^ mid5


MAPPINGS = {
    "Normal": normal_mapping,
    "XOR-A": xor_a_mapping,
    "XOR-B": xor_b_mapping,
}


# ============================================================
# CONFLICT COST
#
# None = inactive lane
# ============================================================

def conflict_degree(addresses, mapping_function):

    counts = {}

    for address in addresses:

        if address is None:
            continue

        bank = mapping_function(address)

        counts[bank] = counts.get(bank, 0) + 1

    if not counts:
        return 0

    return max(counts.values())


def instruction_cost(addresses, mapping_name):

    return conflict_degree(
        addresses,
        MAPPINGS[mapping_name]
    )


def total_cost(stream, mapping_name):

    return sum(
        instruction_cost(
            addresses,
            mapping_name
        )
        for addresses in stream
    )


def window_cost(window, mapping_name):

    return sum(
        instruction_cost(
            addresses,
            mapping_name
        )
        for addresses in window
    )


# ============================================================
# FUTURE WINDOW
# ============================================================

def get_window_result(stream, index, window_size):

    end = min(
        index + window_size,
        len(stream)
    )

    window = stream[index:end]

    costs = {
        mapping_name:
        window_cost(
            window,
            mapping_name
        )
        for mapping_name in MAPPINGS
    }

    winner = min(
        costs,
        key=costs.get
    )

    return winner, costs


# ============================================================
# HYBRID 9/10 + 3/5 SELECTOR
# ============================================================

def choose_hybrid_candidate(
    winner5,
    costs5,
    winner10,
    costs10,
    history5,
    history10
):

    candidate = None
    selected_costs = None


    # --------------------------------------------------------
    # HIGH STABILITY: 9 / 10
    # --------------------------------------------------------

    if len(history10) == 10:

        counts10 = Counter(history10)

        ordered = counts10.most_common()

        best10, votes10 = ordered[0]

        second_votes = (
            ordered[1][1]
            if len(ordered) > 1
            else 0
        )

        if (
            votes10 >= HIGH_STABILITY_VOTES
            and
            votes10 > second_votes
        ):

            candidate = best10
            selected_costs = costs10


    # --------------------------------------------------------
    # MEDIUM STABILITY: 3 / 5
    # --------------------------------------------------------

    if (
        candidate is None
        and
        len(history5) == 5
    ):

        counts5 = Counter(history5)

        best5, votes5 = (
            counts5.most_common(1)[0]
        )

        if votes5 >= MEDIUM_STABILITY_VOTES:

            candidate = best5
            selected_costs = costs5


    # --------------------------------------------------------
    # BASELINE-5
    # --------------------------------------------------------

    if candidate is None:

        candidate = winner5
        selected_costs = costs5


    return candidate, selected_costs


# ============================================================
# BENCHMARK 1
# TILED MATRIX TRANSPOSE
#
# 32x32 unpadded tile.
#
# Row write:
#     tile[row][lane]
#
# Column read:
#     tile[lane][column]
#
# Previous tile is dead when next tile starts.
# ============================================================

def generate_transpose(num_tiles=64):

    TILE_DIM = 32
    TILE_STRIDE = 32

    stream = []
    safe = []


    for tile in range(num_tiles):

        row = tile % TILE_DIM
        column = tile % TILE_DIM


        # ----------------------------------------------------
        # ROW WRITE
        # ----------------------------------------------------

        write_addresses = [
            row * TILE_STRIDE + lane
            for lane in range(WARP_SIZE)
        ]

        stream.append(write_addresses)

        # New tile = old tile dead
        safe.append(True)


        # ----------------------------------------------------
        # COLUMN READ
        # ----------------------------------------------------

        read_addresses = [
            lane * TILE_STRIDE + column
            for lane in range(WARP_SIZE)
        ]

        stream.append(read_addresses)

        safe.append(False)


    return stream, safe


# ============================================================
# BENCHMARK 2
# PARALLEL REDUCTION
#
# Standard contiguous tree reduction.
#
# This should be mostly / fully conflict-free.
#
# Safe remapping only between complete blocks.
# ============================================================

def generate_reduction(
    num_blocks=32,
    block_size=256
):

    stream = []
    safe = []


    for block_id in range(num_blocks):

        first_instruction = True

        stride = block_size // 2


        while stride > 0:

            active_warps = (
                stride
                +
                WARP_SIZE
                -
                1
            ) // WARP_SIZE


            for warp_id in range(
                active_warps
            ):

                warp_base = (
                    warp_id
                    *
                    WARP_SIZE
                )


                self_addresses = []
                partner_addresses = []


                for lane in range(
                    WARP_SIZE
                ):

                    tid = (
                        warp_base
                        +
                        lane
                    )


                    if tid < stride:

                        self_addresses.append(
                            tid
                        )

                        partner_addresses.append(
                            tid + stride
                        )

                    else:

                        self_addresses.append(
                            None
                        )

                        partner_addresses.append(
                            None
                        )


                stream.append(
                    self_addresses
                )

                safe.append(
                    first_instruction
                )

                first_instruction = False


                stream.append(
                    partner_addresses
                )

                safe.append(False)


            stride //= 2


    return stream, safe


# ============================================================
# BENCHMARK 3
# BLELLOCH PREFIX SCAN
#
# Upsweep:
#
# index = (tid + 1) * 2 * offset - 1
#
# accesses:
#     shared[index - offset]
#     shared[index]
#
# Downsweep uses similar accesses.
#
# Shared state remains live throughout one scan block.
#
# Safe remapping only between complete scan blocks.
# ============================================================

def generate_scan(
    num_blocks=32,
    block_size=256
):

    stream = []
    safe = []


    for block_id in range(num_blocks):

        first_instruction = True


        # ====================================================
        # UPSWEEP
        # ====================================================

        offset = 1


        while offset < block_size:

            active_threads = (
                block_size
                //
                (2 * offset)
            )


            active_warps = (

                active_threads
                +
                WARP_SIZE
                -
                1

            ) // WARP_SIZE


            for warp_id in range(
                active_warps
            ):

                left_addresses = []
                right_addresses = []


                for lane in range(
                    WARP_SIZE
                ):

                    tid = (
                        warp_id
                        *
                        WARP_SIZE
                        +
                        lane
                    )


                    if tid < active_threads:

                        index = (
                            (tid + 1)
                            *
                            2
                            *
                            offset
                            -
                            1
                        )


                        left_addresses.append(
                            index - offset
                        )

                        right_addresses.append(
                            index
                        )


                    else:

                        left_addresses.append(
                            None
                        )

                        right_addresses.append(
                            None
                        )


                stream.append(
                    left_addresses
                )

                safe.append(
                    first_instruction
                )

                first_instruction = False


                stream.append(
                    right_addresses
                )

                safe.append(False)


            offset *= 2


        # ====================================================
        # DOWNSWEEP
        # ====================================================

        offset = block_size // 2


        while offset >= 1:

            active_threads = (
                block_size
                //
                (2 * offset)
            )


            active_warps = (

                active_threads
                +
                WARP_SIZE
                -
                1

            ) // WARP_SIZE


            for warp_id in range(
                active_warps
            ):

                left_addresses = []
                right_addresses = []


                for lane in range(
                    WARP_SIZE
                ):

                    tid = (
                        warp_id
                        *
                        WARP_SIZE
                        +
                        lane
                    )


                    if tid < active_threads:

                        index = (
                            (tid + 1)
                            *
                            2
                            *
                            offset
                            -
                            1
                        )


                        left_addresses.append(
                            index - offset
                        )

                        right_addresses.append(
                            index
                        )


                    else:

                        left_addresses.append(
                            None
                        )

                        right_addresses.append(
                            None
                        )


                stream.append(
                    left_addresses
                )

                safe.append(False)


                stream.append(
                    right_addresses
                )

                safe.append(False)


            offset //= 2


    return stream, safe


# ============================================================
# BENCHMARK 4
# PHASE-MIXED SHARED-MEMORY STRESS
#
# This is NOT claimed to be one standard CUDA algorithm.
#
# It combines realistic stride motifs seen in shared-memory
# indexing patterns.
#
# Each region owns fresh shared-memory state, so a new region
# is a legitimate safe remapping point.
#
# Patterns deliberately include:
#
# stride 3  -> Normal preferred
# stride 32 -> XOR-A preferred
# stride 24 -> XOR-B preferred
#
# This benchmark checks whether the adaptive controller can
# exploit phase changes when safe remapping opportunities exist.
# ============================================================

def generate_phase_mixed(
    repetitions=40,
    instructions_per_region=6
):

    stream = []
    safe = []


    phase_strides = [
        3,      # favors Normal
        32,     # favors XOR-A
        24,     # favors XOR-B
        5,      # favors Normal
        16,     # favors XOR-A
        24,     # favors XOR-B
    ]


    for repetition in range(
        repetitions
    ):

        for stride in phase_strides:

            for instruction_id in range(
                instructions_per_region
            ):

                # Slight base variation prevents every
                # instruction from being bit-identical.

                base = (
                    repetition * 97
                    +
                    instruction_id * 11
                )


                addresses = [

                    base
                    +
                    lane * stride

                    for lane
                    in range(WARP_SIZE)
                ]


                stream.append(
                    addresses
                )


                safe.append(
                    instruction_id == 0
                )


    return stream, safe


# ============================================================
# EXP-29
# UNRESTRICTED CONTROLLER
#
# Full lookahead is assumed available for these
# deterministic validation traces.
# ============================================================

def run_exp29(stream):

    current = INITIAL_MAPPING

    history5 = deque(maxlen=5)
    history10 = deque(maxlen=10)

    watchdog_history = deque(
        maxlen=WATCHDOG_WINDOW
    )

    since_switch = MIN_HOLD
    recovery_lock = 0

    cost = 0
    switches = 0

    mapping_trace = []


    for i, addresses in enumerate(
        stream
    ):

        watchdog_fired = False


        # ====================================================
        # CAUSAL WATCHDOG
        # ====================================================

        if (
            len(watchdog_history)
            ==
            WATCHDOG_WINDOW
        ):

            previous = list(
                watchdog_history
            )


            wd_costs = {

                name:
                window_cost(
                    previous,
                    name
                )

                for name in MAPPINGS
            }


            best = min(
                wd_costs,
                key=wd_costs.get
            )


            if best != current:

                current_cost = (
                    wd_costs[current]
                )


                if current_cost > 0:

                    loss = (

                        current_cost
                        -
                        wd_costs[best]

                    ) / current_cost


                    if (
                        loss
                        >=
                        WATCHDOG_THRESHOLD
                    ):

                        current = best

                        switches += 1

                        since_switch = 0

                        recovery_lock = (
                            RECOVERY_LOCK
                        )

                        watchdog_fired = True


        # ====================================================
        # LOOKAHEAD
        # ====================================================

        if not watchdog_fired:

            winner5, costs5 = (
                get_window_result(
                    stream,
                    i,
                    5
                )
            )


            winner10, costs10 = (
                get_window_result(
                    stream,
                    i,
                    10
                )
            )


            history5.append(
                winner5
            )

            history10.append(
                winner10
            )


            emergency_fired = False


            # =================================================
            # EMERGENCY BYPASS
            # =================================================

            if winner5 != current:

                current_cost = (
                    costs5[current]
                )


                if current_cost > 0:

                    gain = (

                        current_cost
                        -
                        costs5[
                            winner5
                        ]

                    ) / current_cost


                    if (
                        gain
                        >=
                        EMERGENCY_THRESHOLD
                        and
                        since_switch
                        >=
                        MIN_HOLD
                    ):

                        current = winner5

                        switches += 1

                        since_switch = 0

                        emergency_fired = True


            # =================================================
            # NORMAL HYBRID
            # =================================================

            if (
                not emergency_fired
                and
                recovery_lock == 0
            ):

                (
                    candidate,
                    selected_costs
                ) = choose_hybrid_candidate(
                    winner5,
                    costs5,
                    winner10,
                    costs10,
                    history5,
                    history10
                )


                if candidate != current:

                    current_cost = (
                        selected_costs[
                            current
                        ]
                    )


                    if current_cost > 0:

                        improvement = (

                            current_cost
                            -
                            selected_costs[
                                candidate
                            ]

                        ) / current_cost


                        if (
                            since_switch
                            >=
                            MIN_HOLD
                            and
                            improvement
                            >=
                            SWITCH_THRESHOLD
                        ):

                            current = candidate

                            switches += 1

                            since_switch = 0


        mapping_trace.append(
            current
        )


        cost += (
            instruction_cost(
                addresses,
                current
            )
        )


        since_switch += 1


        if recovery_lock > 0:

            recovery_lock -= 1


        watchdog_history.append(
            addresses
        )


    return {
        "cost": cost,
        "switches": switches,
        "trace": mapping_trace,
    }


# ============================================================
# EXP-35
# SAFE SWITCHING + DEFER + REVALIDATION
# ============================================================

def run_exp35(
    stream,
    safe_boundaries
):

    current = INITIAL_MAPPING

    history5 = deque(maxlen=5)
    history10 = deque(maxlen=10)

    watchdog_history = deque(
        maxlen=WATCHDOG_WINDOW
    )


    since_switch = MIN_HOLD
    recovery_lock = 0


    cost = 0
    switches = 0

    blocked = 0

    pending_mapping = None
    pending_source = None

    revalidation_attempts = 0
    revalidation_applied = 0
    revalidation_rejected = 0
    stale_rejected = 0

    mapping_trace = []


    def store_pending(
        mapping,
        source
    ):

        nonlocal pending_mapping
        nonlocal pending_source

        pending_mapping = mapping
        pending_source = source


    for i, addresses in enumerate(
        stream
    ):

        safe = (
            safe_boundaries[i]
        )


        revalidation_switched = False


        # ====================================================
        # REVALIDATE PENDING REQUEST
        # ====================================================

        if (
            safe
            and
            pending_mapping is not None
        ):

            revalidation_attempts += 1


            candidate = (
                pending_mapping
            )

            source = (
                pending_source
            )


            valid = False


            # ------------------------------------------------
            # WATCHDOG REQUEST
            # ------------------------------------------------

            if source == "watchdog":

                if (
                    len(watchdog_history)
                    ==
                    WATCHDOG_WINDOW
                ):

                    previous = list(
                        watchdog_history
                    )


                    costs = {

                        name:
                        window_cost(
                            previous,
                            name
                        )

                        for name
                        in MAPPINGS
                    }


                    best = min(
                        costs,
                        key=costs.get
                    )


                    if (
                        best == candidate
                        and
                        candidate != current
                    ):

                        current_cost = (
                            costs[current]
                        )


                        if current_cost > 0:

                            loss = (

                                current_cost
                                -
                                costs[
                                    candidate
                                ]

                            ) / current_cost


                            if (
                                loss
                                >=
                                WATCHDOG_THRESHOLD
                            ):

                                valid = True


            # ------------------------------------------------
            # EMERGENCY REQUEST
            # ------------------------------------------------

            elif source == "emergency":

                winner5, costs5 = (
                    get_window_result(
                        stream,
                        i,
                        5
                    )
                )


                if (
                    winner5 == candidate
                    and
                    candidate != current
                ):

                    current_cost = (
                        costs5[current]
                    )


                    if current_cost > 0:

                        gain = (

                            current_cost
                            -
                            costs5[
                                candidate
                            ]

                        ) / current_cost


                        if (
                            gain
                            >=
                            EMERGENCY_THRESHOLD
                        ):

                            valid = True


            # ------------------------------------------------
            # NORMAL REQUEST
            # ------------------------------------------------

            elif source == "normal":

                winner5, costs5 = (
                    get_window_result(
                        stream,
                        i,
                        5
                    )
                )


                winner10, costs10 = (
                    get_window_result(
                        stream,
                        i,
                        10
                    )
                )


                temp5 = deque(
                    history5,
                    maxlen=5
                )

                temp10 = deque(
                    history10,
                    maxlen=10
                )


                temp5.append(
                    winner5
                )

                temp10.append(
                    winner10
                )


                (
                    new_candidate,
                    selected_costs
                ) = choose_hybrid_candidate(
                    winner5,
                    costs5,
                    winner10,
                    costs10,
                    temp5,
                    temp10
                )


                if (
                    new_candidate
                    ==
                    candidate
                    and
                    candidate
                    !=
                    current
                ):

                    current_cost = (
                        selected_costs[
                            current
                        ]
                    )


                    if current_cost > 0:

                        gain = (

                            current_cost
                            -
                            selected_costs[
                                candidate
                            ]

                        ) / current_cost


                        if (
                            gain
                            >=
                            SWITCH_THRESHOLD
                        ):

                            valid = True


            # ------------------------------------------------
            # APPLY
            # ------------------------------------------------

            if (
                valid
                and
                since_switch
                >=
                MIN_HOLD
            ):

                current = candidate

                switches += 1

                revalidation_applied += 1

                since_switch = 0


                if source == "watchdog":

                    recovery_lock = (
                        RECOVERY_LOCK
                    )


                revalidation_switched = True


            else:

                revalidation_rejected += 1
                stale_rejected += 1


            pending_mapping = None
            pending_source = None


        watchdog_fired = False


        # ====================================================
        # CAUSAL WATCHDOG
        # ====================================================

        if (
            not revalidation_switched
            and
            len(watchdog_history)
            ==
            WATCHDOG_WINDOW
        ):

            previous = list(
                watchdog_history
            )


            costs = {

                name:
                window_cost(
                    previous,
                    name
                )

                for name
                in MAPPINGS
            }


            best = min(
                costs,
                key=costs.get
            )


            if best != current:

                current_cost = (
                    costs[current]
                )


                if current_cost > 0:

                    loss = (

                        current_cost
                        -
                        costs[best]

                    ) / current_cost


                    if (
                        loss
                        >=
                        WATCHDOG_THRESHOLD
                    ):

                        if safe:

                            current = best

                            switches += 1

                            since_switch = 0

                            recovery_lock = (
                                RECOVERY_LOCK
                            )

                            watchdog_fired = True


                        else:

                            blocked += 1

                            store_pending(
                                best,
                                "watchdog"
                            )


        # ====================================================
        # LOOKAHEAD
        # ====================================================

        if (
            not revalidation_switched
            and
            not watchdog_fired
        ):

            winner5, costs5 = (
                get_window_result(
                    stream,
                    i,
                    5
                )
            )


            winner10, costs10 = (
                get_window_result(
                    stream,
                    i,
                    10
                )
            )


            history5.append(
                winner5
            )

            history10.append(
                winner10
            )


            emergency_fired = False


            # =================================================
            # EMERGENCY
            # =================================================

            if winner5 != current:

                current_cost = (
                    costs5[current]
                )


                if current_cost > 0:

                    gain = (

                        current_cost
                        -
                        costs5[
                            winner5
                        ]

                    ) / current_cost


                    if (
                        gain
                        >=
                        EMERGENCY_THRESHOLD
                        and
                        since_switch
                        >=
                        MIN_HOLD
                    ):

                        if safe:

                            current = winner5

                            switches += 1

                            since_switch = 0

                            emergency_fired = True


                        else:

                            blocked += 1

                            store_pending(
                                winner5,
                                "emergency"
                            )


            # =================================================
            # NORMAL HYBRID
            # =================================================

            if (
                not emergency_fired
                and
                recovery_lock == 0
            ):

                (
                    candidate,
                    selected_costs
                ) = choose_hybrid_candidate(
                    winner5,
                    costs5,
                    winner10,
                    costs10,
                    history5,
                    history10
                )


                if candidate != current:

                    current_cost = (
                        selected_costs[
                            current
                        ]
                    )


                    if current_cost > 0:

                        gain = (

                            current_cost
                            -
                            selected_costs[
                                candidate
                            ]

                        ) / current_cost


                        if (
                            since_switch
                            >=
                            MIN_HOLD
                            and
                            gain
                            >=
                            SWITCH_THRESHOLD
                        ):

                            if safe:

                                current = candidate

                                switches += 1

                                since_switch = 0


                            else:

                                blocked += 1

                                store_pending(
                                    candidate,
                                    "normal"
                                )


        mapping_trace.append(
            current
        )


        cost += (
            instruction_cost(
                addresses,
                current
            )
        )


        since_switch += 1


        if recovery_lock > 0:

            recovery_lock -= 1


        watchdog_history.append(
            addresses
        )


    return {
        "cost": cost,
        "switches": switches,
        "blocked": blocked,
        "revalidation_attempts":
            revalidation_attempts,
        "revalidation_applied":
            revalidation_applied,
        "revalidation_rejected":
            revalidation_rejected,
        "stale_rejected":
            stale_rejected,
        "trace":
            mapping_trace,
    }


# ============================================================
# SAFE-SWITCH CORRECTNESS CHECK
# ============================================================

def count_unsafe_switches(
    trace,
    safe_boundaries
):

    unsafe = 0


    for i in range(
        1,
        len(trace)
    ):

        if (
            trace[i]
            !=
            trace[i - 1]
            and
            not safe_boundaries[i]
        ):

            unsafe += 1


    return unsafe


# ============================================================
# BENCHMARK ANALYSIS
# ============================================================

def analyze_benchmark(
    name,
    stream,
    safe_boundaries
):

    static = {

        mapping:
        total_cost(
            stream,
            mapping
        )

        for mapping
        in MAPPINGS
    }


    best_static_mapping = min(
        static,
        key=static.get
    )


    best_static_cost = (
        static[
            best_static_mapping
        ]
    )


    exp29 = run_exp29(
        stream
    )


    exp35 = run_exp35(
        stream,
        safe_boundaries
    )


    unsafe = (
        count_unsafe_switches(
            exp35["trace"],
            safe_boundaries
        )
    )


    # ========================================================
    # IMPROVEMENT AGAINST BEST STATIC
    # ========================================================

    exp29_vs_static = (

        best_static_cost
        -
        exp29["cost"]

    ) / best_static_cost * 100


    exp35_vs_static = (

        best_static_cost
        -
        exp35["cost"]

    ) / best_static_cost * 100


    # ========================================================
    # RETENTION
    #
    # Compare adaptive advantage relative to best static.
    # ========================================================

    gain29 = (
        best_static_cost
        -
        exp29["cost"]
    )

    gain35 = (
        best_static_cost
        -
        exp35["cost"]
    )


    if gain29 > 0:

        retention = (
            gain35
            /
            gain29
            *
            100
        )

    elif exp35["cost"] == exp29["cost"]:

        retention = 100.0

    else:

        retention = 0.0


    density = (
        sum(safe_boundaries)
        /
        len(safe_boundaries)
        *
        100
    )


    return {

        "name":
            name,

        "instructions":
            len(stream),

        "safe_density":
            density,

        "normal":
            static["Normal"],

        "xor_a":
            static["XOR-A"],

        "xor_b":
            static["XOR-B"],

        "best_static":
            best_static_mapping,

        "best_static_cost":
            best_static_cost,

        "exp29_cost":
            exp29["cost"],

        "exp29_switches":
            exp29["switches"],

        "exp29_vs_static":
            exp29_vs_static,

        "exp35_cost":
            exp35["cost"],

        "exp35_switches":
            exp35["switches"],

        "exp35_vs_static":
            exp35_vs_static,

        "blocked":
            exp35["blocked"],

        "rv_attempts":
            exp35[
                "revalidation_attempts"
            ],

        "rv_applied":
            exp35[
                "revalidation_applied"
            ],

        "rv_rejected":
            exp35[
                "revalidation_rejected"
            ],

        "stale":
            exp35[
                "stale_rejected"
            ],

        "unsafe":
            unsafe,

        "retention":
            retention,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 120
    )

    print(
        "FINAL REALISM VALIDATION"
    )

    print(
        "=" * 120
    )

    print(
        "Mappings: Normal / XOR-A / XOR-B"
    )

    print(
        "Controllers: Exp-29 unrestricted / "
        "Exp-35 lifetime-safe + revalidation"
    )

    print()


    benchmarks = []


    # ========================================================
    # TRANSPOSE
    # ========================================================

    stream, safe = (
        generate_transpose()
    )

    benchmarks.append(
        analyze_benchmark(
            "Transpose",
            stream,
            safe
        )
    )


    # ========================================================
    # REDUCTION
    # ========================================================

    stream, safe = (
        generate_reduction()
    )

    benchmarks.append(
        analyze_benchmark(
            "Reduction",
            stream,
            safe
        )
    )


    # ========================================================
    # SCAN
    # ========================================================

    stream, safe = (
        generate_scan()
    )

    benchmarks.append(
        analyze_benchmark(
            "Prefix Scan",
            stream,
            safe
        )
    )


    # ========================================================
    # PHASE-MIXED
    # ========================================================

    stream, safe = (
        generate_phase_mixed()
    )

    benchmarks.append(
        analyze_benchmark(
            "Phase-Mixed",
            stream,
            safe
        )
    )


    # ========================================================
    # STATIC RESULTS
    # ========================================================

    print(
        "=" * 120
    )

    print(
        "STATIC MAPPING RESULTS"
    )

    print(
        "=" * 120
    )


    print(
        f"{'Benchmark':<16}"
        f"{'Normal':>12}"
        f"{'XOR-A':>12}"
        f"{'XOR-B':>12}"
        f"{'Best':>12}"
        f"{'BestCost':>12}"
        f"{'Safe%':>10}"
    )

    print(
        "-" * 120
    )


    for r in benchmarks:

        print(

            f"{r['name']:<16}"

            f"{r['normal']:>12}"

            f"{r['xor_a']:>12}"

            f"{r['xor_b']:>12}"

            f"{r['best_static']:>12}"

            f"{r['best_static_cost']:>12}"

            f"{r['safe_density']:>9.2f}%"
        )


    # ========================================================
    # ADAPTIVE RESULTS
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "ADAPTIVE CONTROLLER RESULTS"
    )

    print(
        "=" * 120
    )


    print(
        f"{'Benchmark':<16}"
        f"{'Static':>10}"
        f"{'Exp29':>10}"
        f"{'E29Sw':>8}"
        f"{'E29 vs S':>11}"
        f"{'Exp35':>10}"
        f"{'E35Sw':>8}"
        f"{'E35 vs S':>11}"
        f"{'Unsafe':>9}"
    )

    print(
        "-" * 120
    )


    for r in benchmarks:

        print(

            f"{r['name']:<16}"

            f"{r['best_static_cost']:>10}"

            f"{r['exp29_cost']:>10}"

            f"{r['exp29_switches']:>8}"

            f"{r['exp29_vs_static']:>10.2f}%"

            f"{r['exp35_cost']:>10}"

            f"{r['exp35_switches']:>8}"

            f"{r['exp35_vs_static']:>10.2f}%"

            f"{r['unsafe']:>9}"
        )


    # ========================================================
    # EXP-35 DETAILS
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "EXP-35 SAFE-REMAPPING DETAILS"
    )

    print(
        "=" * 120
    )


    print(
        f"{'Benchmark':<16}"
        f"{'Blocked':>10}"
        f"{'RV Try':>10}"
        f"{'RV Apply':>10}"
        f"{'RV Reject':>11}"
        f"{'Stale':>10}"
        f"{'Retention':>12}"
        f"{'Correct':>10}"
    )

    print(
        "-" * 120
    )


    for r in benchmarks:

        correctness = (
            "PASS"
            if r["unsafe"] == 0
            else "FAIL"
        )


        print(

            f"{r['name']:<16}"

            f"{r['blocked']:>10}"

            f"{r['rv_attempts']:>10}"

            f"{r['rv_applied']:>10}"

            f"{r['rv_rejected']:>11}"

            f"{r['stale']:>10}"

            f"{r['retention']:>11.2f}%"

            f"{correctness:>10}"
        )


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()

    print(
        "=" * 120
    )

    print(
        "BENCHMARK INTERPRETATION"
    )

    print(
        "=" * 120
    )


    for r in benchmarks:

        print()

        print(
            r["name"]
        )

        print(
            f"  Best static mapping: "
            f"{r['best_static']}"
        )

        print(
            f"  Best static cost: "
            f"{r['best_static_cost']}"
        )

        print(
            f"  Exp-29 cost: "
            f"{r['exp29_cost']}"
        )

        print(
            f"  Exp-35 cost: "
            f"{r['exp35_cost']}"
        )

        print(
            f"  Safe-boundary density: "
            f"{r['safe_density']:.2f}%"
        )

        print(
            f"  Unsafe switches: "
            f"{r['unsafe']}"
        )


        if (
            r["exp35_cost"]
            <
            r["best_static_cost"]
        ):

            print(
                "  RESULT: Exp-35 beats "
                "the best static mapping."
            )


        elif (
            r["exp35_cost"]
            ==
            r["best_static_cost"]
        ):

            print(
                "  RESULT: Exp-35 matches "
                "the best static mapping."
            )


        else:

            print(
                "  RESULT: Exp-35 is worse "
                "than the best static mapping."
            )


    print()

    print(
        "=" * 120
    )

    print(
        "IMPORTANT REPORTING NOTE"
    )

    print(
        "=" * 120
    )

    print(
        "Transpose, Reduction and Prefix Scan are "
        "kernel-derived algorithmic traces."
    )

    print(
        "Phase-Mixed is a synthetic stress benchmark composed "
        "of realistic shared-memory stride motifs."
    )

    print(
        "Do NOT describe these results as hardware-measured "
        "GPU execution time or NVIDIA hardware traces."
    )