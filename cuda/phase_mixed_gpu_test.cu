#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

#define THREADS 32
#define BLOCKS 256
#define SHARED_SIZE 1024

#define PHASE_COUNT 11
#define PHASE_REPETITIONS 100
#define INNER_ITERATIONS 100

__device__ __forceinline__
int phase_stride(int phase)
{
    const int strides[PHASE_COUNT] = {
        1, 2, 4, 8, 16, 32,
        16, 8, 4, 2, 1
    };

    return strides[phase];
}

__global__
void conflict_free_reference(int *output)
{
    __shared__ volatile int shared[SHARED_SIZE];

    int tid = threadIdx.x;

    shared[tid] = tid + 1;

    __syncthreads();

    int value = tid + 1;

    for (int repetition = 0;
         repetition < PHASE_REPETITIONS;
         repetition++)
    {
        for (int phase = 0;
             phase < PHASE_COUNT;
             phase++)
        {
            int index = tid;

            #pragma unroll 1
            for (int i = 0;
                 i < INNER_ITERATIONS;
                 i++)
            {
                value += shared[index];
                shared[index] = value;
            }

            __syncthreads();
        }
    }

    output[blockIdx.x * blockDim.x + tid] = value;
}

__global__
void phase_mixed_conflicts(int *output)
{
    __shared__ volatile int shared[SHARED_SIZE];

    int tid = threadIdx.x;

    for (int i = tid;
         i < SHARED_SIZE;
         i += blockDim.x)
    {
        shared[i] = i + 1;
    }

    __syncthreads();

    int value = tid + 1;

    for (int repetition = 0;
         repetition < PHASE_REPETITIONS;
         repetition++)
    {
        for (int phase = 0;
             phase < PHASE_COUNT;
             phase++)
        {
            int stride = phase_stride(phase);

            int index = tid * stride;

            #pragma unroll 1
            for (int i = 0;
                 i < INNER_ITERATIONS;
                 i++)
            {
                value += shared[index];
                shared[index] = value;
            }

            __syncthreads();
        }
    }

    output[blockIdx.x * blockDim.x + tid] = value;
}

float run_reference(int *d_output)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    conflict_free_reference
        <<<BLOCKS, THREADS>>>(
            d_output
        );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

float run_phase_mixed(int *d_output)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    phase_mixed_conflicts
        <<<BLOCKS, THREADS>>>(
            d_output
        );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

double calculate_mean(
    const std::vector<float>& values
)
{
    return std::accumulate(
        values.begin(),
        values.end(),
        0.0
    ) / values.size();
}

double calculate_stddev(
    const std::vector<float>& values,
    double mean
)
{
    double sum = 0.0;

    for (float value : values)
    {
        double difference =
            value - mean;

        sum += difference * difference;
    }

    return std::sqrt(
        sum / values.size()
    );
}

double calculate_median(
    std::vector<float> values
)
{
    std::sort(
        values.begin(),
        values.end()
    );

    size_t size =
        values.size();

    if (size % 2 == 0)
    {
        return (
            values[size / 2 - 1]
            +
            values[size / 2]
        ) / 2.0;
    }

    return values[size / 2];
}

void print_statistics(
    const std::string& name,
    const std::vector<float>& values
)
{
    double mean =
        calculate_mean(values);

    double median =
        calculate_median(values);

    double stddev =
        calculate_stddev(
            values,
            mean
        );

    float minimum =
        *std::min_element(
            values.begin(),
            values.end()
        );

    float maximum =
        *std::max_element(
            values.begin(),
            values.end()
        );

    std::cout
        << "\n"
        << name
        << "\n";

    std::cout
        << "Mean     : "
        << std::fixed
        << std::setprecision(4)
        << mean
        << " ms\n";

    std::cout
        << "Median   : "
        << median
        << " ms\n";

    std::cout
        << "Std Dev  : "
        << stddev
        << " ms\n";

    std::cout
        << "Minimum  : "
        << minimum
        << " ms\n";

    std::cout
        << "Maximum  : "
        << maximum
        << " ms\n";
}

int main()
{
    const int RUNS = 50;
    const int WARMUP_RUNS = 10;

    int *d_output;

    cudaMalloc(
        &d_output,
        BLOCKS
        *
        THREADS
        *
        sizeof(int)
    );

    std::cout
        << "=============================================\n";

    std::cout
        << "PHASE-CHANGING SHARED-MEMORY BANK-CONFLICT\n";

    std::cout
        << "REAL GPU STRESS TEST\n";

    std::cout
        << "=============================================\n";

    std::cout
        << "Blocks              : "
        << BLOCKS
        << "\n";

    std::cout
        << "Threads/block       : "
        << THREADS
        << "\n";

    std::cout
        << "Measured runs       : "
        << RUNS
        << "\n";

    std::cout
        << "Warm-up runs        : "
        << WARMUP_RUNS
        << "\n";

    std::cout
        << "Phase repetitions   : "
        << PHASE_REPETITIONS
        << "\n";

    std::cout
        << "Iterations/phase    : "
        << INNER_ITERATIONS
        << "\n";

    std::cout
        << "Conflict sequence   : "
        << "1 2 4 8 16 32 16 8 4 2 1\n";

    // WARM-UP

    for (int i = 0;
         i < WARMUP_RUNS;
         i++)
    {
        run_reference(
            d_output
        );

        run_phase_mixed(
            d_output
        );
    }

    std::vector<float>
        reference_times;

    std::vector<float>
        phase_times;

    // MEASURE

    for (int i = 0;
         i < RUNS;
         i++)
    {
        reference_times.push_back(
            run_reference(
                d_output
            )
        );

        phase_times.push_back(
            run_phase_mixed(
                d_output
            )
        );
    }

    cudaDeviceSynchronize();

    // RESULTS

    print_statistics(
        "CONFLICT-FREE REFERENCE",
        reference_times
    );

    print_statistics(
        "PHASE-MIXED BANK CONFLICTS",
        phase_times
    );

    double reference_mean =
        calculate_mean(
            reference_times
        );

    double phase_mean =
        calculate_mean(
            phase_times
        );

    double slowdown =
        phase_mean /
        reference_mean;

    double overhead =
        (
            phase_mean
            -
            reference_mean
        )
        /
        reference_mean
        *
        100.0;

    std::cout
        << "\n=============================================\n";

    std::cout
        << "FINAL HARDWARE RESULT\n";

    std::cout
        << "=============================================\n";

    std::cout
        << "Conflict-free mean : "
        << reference_mean
        << " ms\n";

    std::cout
        << "Phase-mixed mean   : "
        << phase_mean
        << " ms\n";

    std::cout
        << "Runtime slowdown   : "
        << slowdown
        << "x\n";

    std::cout
        << "Runtime overhead   : "
        << overhead
        << "%\n";

    std::cout
        << "=============================================\n";

    cudaFree(
        d_output
    );

    return 0;
}