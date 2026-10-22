#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

#define TILE 32
#define MATRIX_SIZE 4096
#define WARMUPS 10
#define RUNS 50

// Baseline: 32 x 32 shared-memory tile.
// During transposed column access, threads can hit the same bank.
__global__ void transpose_baseline(const float* input, float* output)
{
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < MATRIX_SIZE && y < MATRIX_SIZE)
        tile[threadIdx.y][threadIdx.x] =
            input[y * MATRIX_SIZE + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < MATRIX_SIZE && y < MATRIX_SIZE)
        output[y * MATRIX_SIZE + x] =
            tile[threadIdx.x][threadIdx.y];
}

// Optimized: exact same transpose, but one padding element
// changes the shared-memory bank placement.
__global__ void transpose_padded(const float* input, float* output)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < MATRIX_SIZE && y < MATRIX_SIZE)
        tile[threadIdx.y][threadIdx.x] =
            input[y * MATRIX_SIZE + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < MATRIX_SIZE && y < MATRIX_SIZE)
        output[y * MATRIX_SIZE + x] =
            tile[threadIdx.x][threadIdx.y];
}

float run_baseline(
    const float* d_input,
    float* d_output,
    dim3 grid,
    dim3 block)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    transpose_baseline<<<grid, block>>>(
        d_input,
        d_output
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float run_padded(
    const float* d_input,
    float* d_output,
    dim3 grid,
    dim3 block)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    transpose_padded<<<grid, block>>>(
        d_input,
        d_output
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

void calculate_stats(
    const std::vector<float>& values,
    double& mean,
    double& median,
    double& sd,
    float& minimum,
    float& maximum)
{
    mean =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        ) / values.size();

    std::vector<float> temp = values;
    std::sort(temp.begin(), temp.end());

    median =
        (temp[temp.size() / 2 - 1] +
         temp[temp.size() / 2]) / 2.0;

    double variance = 0.0;

    for (float x : values)
        variance += (x - mean) * (x - mean);

    variance /= values.size();

    sd = std::sqrt(variance);

    minimum =
        *std::min_element(
            values.begin(),
            values.end()
        );

    maximum =
        *std::max_element(
            values.begin(),
            values.end()
        );
}

int main()
{
    size_t elements =
        (size_t)MATRIX_SIZE * MATRIX_SIZE;

    size_t bytes =
        elements * sizeof(float);

    std::vector<float> h_input(elements);
    std::vector<float> h_baseline(elements);
    std::vector<float> h_padded(elements);

    for (size_t i = 0; i < elements; i++)
        h_input[i] = (float)(i % 1000);

    float* d_input;
    float* d_baseline;
    float* d_padded;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_baseline, bytes);
    cudaMalloc(&d_padded, bytes);

    cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    dim3 block(TILE, TILE);

    dim3 grid(
        (MATRIX_SIZE + TILE - 1) / TILE,
        (MATRIX_SIZE + TILE - 1) / TILE
    );

    std::cout
        << "=============================================\n"
        << "FAIR SHARED-MEMORY BANK-CONFLICT A/B TEST\n"
        << "=============================================\n";

    std::cout
        << "GPU matrix size      : "
        << MATRIX_SIZE << " x "
        << MATRIX_SIZE << "\n";

    std::cout
        << "Tile size            : "
        << TILE << " x "
        << TILE << "\n";

    std::cout
        << "Warm-up runs         : "
        << WARMUPS << "\n";

    std::cout
        << "Measured runs        : "
        << RUNS << "\n\n";

    // Warm-up
    for (int i = 0; i < WARMUPS; i++)
    {
        run_baseline(
            d_input,
            d_baseline,
            grid,
            block
        );

        run_padded(
            d_input,
            d_padded,
            grid,
            block
        );
    }

    std::vector<float> baseline_times;
    std::vector<float> padded_times;

    // Alternate order to reduce systematic bias.
    for (int i = 0; i < RUNS; i++)
    {
        if (i % 2 == 0)
        {
            baseline_times.push_back(
                run_baseline(
                    d_input,
                    d_baseline,
                    grid,
                    block
                )
            );

            padded_times.push_back(
                run_padded(
                    d_input,
                    d_padded,
                    grid,
                    block
                )
            );
        }
        else
        {
            float padded =
                run_padded(
                    d_input,
                    d_padded,
                    grid,
                    block
                );

            float baseline =
                run_baseline(
                    d_input,
                    d_baseline,
                    grid,
                    block
                );

            baseline_times.push_back(baseline);
            padded_times.push_back(padded);
        }
    }

    cudaMemcpy(
        h_baseline.data(),
        d_baseline,
        bytes,
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        h_padded.data(),
        d_padded,
        bytes,
        cudaMemcpyDeviceToHost
    );

    // Verify that both kernels perform the same computation.
    size_t errors = 0;

    for (size_t i = 0; i < elements; i++)
    {
        if (h_baseline[i] != h_padded[i])
        {
            errors++;

            if (errors <= 5)
            {
                std::cout
                    << "Mismatch at "
                    << i << ": "
                    << h_baseline[i]
                    << " vs "
                    << h_padded[i]
                    << "\n";
            }
        }
    }

    double bmean, bmedian, bsd;
    double pmean, pmedian, psd;

    float bmin, bmax;
    float pmin, pmax;

    calculate_stats(
        baseline_times,
        bmean,
        bmedian,
        bsd,
        bmin,
        bmax
    );

    calculate_stats(
        padded_times,
        pmean,
        pmedian,
        psd,
        pmin,
        pmax
    );

    double reduction =
        ((bmean - pmean) / bmean) * 100.0;

    double speedup =
        bmean / pmean;

    std::cout
        << std::fixed
        << std::setprecision(4);

    std::cout
        << "\nBASELINE (UNPADDED)\n";

    std::cout << "Mean     : " << bmean << " ms\n";
    std::cout << "Median   : " << bmedian << " ms\n";
    std::cout << "Std Dev  : " << bsd << " ms\n";
    std::cout << "Minimum  : " << bmin << " ms\n";
    std::cout << "Maximum  : " << bmax << " ms\n";

    std::cout
        << "\nCONFLICT-REDUCED (PADDED)\n";

    std::cout << "Mean     : " << pmean << " ms\n";
    std::cout << "Median   : " << pmedian << " ms\n";
    std::cout << "Std Dev  : " << psd << " ms\n";
    std::cout << "Minimum  : " << pmin << " ms\n";
    std::cout << "Maximum  : " << pmax << " ms\n";

    std::cout
        << "\n=============================================\n"
        << "CORRECTNESS\n"
        << "=============================================\n";

    std::cout
        << "Output mismatches   : "
        << errors << "\n";

    std::cout
        << "Status              : "
        << (errors == 0 ? "PASS" : "FAIL")
        << "\n";

    std::cout
        << "\n=============================================\n"
        << "REAL HARDWARE RESULT\n"
        << "=============================================\n";

    std::cout
        << "Baseline mean       : "
        << bmean << " ms\n";

    std::cout
        << "Padded mean         : "
        << pmean << " ms\n";

    std::cout
        << "Runtime reduction   : "
        << reduction << "%\n";

    std::cout
        << "Speedup             : "
        << speedup << "x\n";

    std::cout
        << "=============================================\n";

    cudaFree(d_input);
    cudaFree(d_baseline);
    cudaFree(d_padded);

    return 0;
}