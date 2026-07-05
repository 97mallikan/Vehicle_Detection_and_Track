// motion_kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <math.h>

extern "C" {

struct RegionStatGPU {
    int area;
    int minx, miny, maxx, maxy;
    float sumx, sumy;
};

} // extern "C"


// ============================================================
// Helpers
// ============================================================
__device__ inline int sample_idx(
    int s, int y, int x, int c,
    int height, int width, int channels
) {
    // Layout: [history][H][W][C]
    return (((s * height + y) * width + x) * channels + c);
}

__device__ inline unsigned int xorshift32(unsigned int x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__device__ inline float rand01(unsigned int seed) {
    seed = xorshift32(seed);
    return (seed & 0x00FFFFFF) / 16777216.0f;
}

__device__ inline int angleToBin(float vx, float vy, int numBins) {
    float mag = sqrtf(vx * vx + vy * vy);
    if (mag < 1e-6f) return -1;

    float ang = atan2f(vy, vx) * 180.0f / 3.14159265358979323846f;
    if (ang < 0.0f) ang += 360.0f;

    float binSize = 360.0f / (float)numBins;
    int b = (int)floorf((ang + 0.5f * binSize) / binSize) % numBins;
    return b;
}


// ============================================================
// Custom CUDA KNN background model
// ============================================================
__global__ void knnInitKernel(
    unsigned char* d_samples,
    unsigned short* d_ages,
    int width,
    int height,
    int history,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Initialize sample slots to 0 and age to a large value
    for (int s = 0; s < history; ++s) {
        for (int c = 0; c < channels; ++c) {
            int idx = sample_idx(s, y, x, c, height, width, channels);
            d_samples[idx] = 0;
        }
        d_ages[s * height * width + y * width + x] = 65535;
    }
}


__global__ void knnApplyKernel(
    const unsigned char* d_frame,   // BGR frame [H][W][C]
    unsigned char* d_fgmask,        // output [H][W]
    unsigned char* d_samples,       // [history][H][W][C]
    unsigned short* d_ages,         // [history][H][W]
    int width,
    int height,
    int history,
    int channels,
    float dist2Threshold,
    int requiredMatches,
    float replaceProb,
    unsigned int frameSeed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int pixBase = (y * width + x) * channels;

    int matchCount = 0;
    int oldestSlot = 0;
    unsigned short oldestAge = 0;

    // Check all history samples
    for (int s = 0; s < history; ++s) {
        float d2 = 0.0f;

        for (int c = 0; c < channels; ++c) {
            int idx = sample_idx(s, y, x, c, height, width, channels);
            float diff = float(d_frame[pixBase + c]) - float(d_samples[idx]);
            d2 += diff * diff;
        }

        unsigned short age = d_ages[s * height * width + y * width + x];

        if (s == 0 || age > oldestAge) {
            oldestAge = age;
            oldestSlot = s;
        }

        if (d2 < dist2Threshold) {
            matchCount++;
        }
    }

    const bool isBackground = (matchCount >= requiredMatches);
    d_fgmask[y * width + x] = isBackground ? 0 : 255;

    // Age all samples at this pixel
    for (int s = 0; s < history; ++s) {
        unsigned short& a = d_ages[s * height * width + y * width + x];
        if (a < 65535) a++;
    }

    // Random replacement of oldest slot
    unsigned int seed = frameSeed ^ (unsigned int)(y * width + x + 1) * 9781u;
    float r = rand01(seed);

    if (r < replaceProb) {
        for (int c = 0; c < channels; ++c) {
            int idx = sample_idx(oldestSlot, y, x, c, height, width, channels);
            d_samples[idx] = d_frame[pixBase + c];
        }
        d_ages[oldestSlot * height * width + y * width + x] = 0;
    }

    // Optional neighbor-style update imitation:
    // if background, sometimes refresh a second random slot.
    if (isBackground) {
        float r2 = rand01(seed ^ 0xA341316Cu);
        if (r2 < 0.5f * replaceProb) {
            int slot2 = (int)(rand01(seed ^ 0xC8013EA4u) * history);
            if (slot2 >= history) slot2 = history - 1;
            for (int c = 0; c < channels; ++c) {
                int idx = sample_idx(slot2, y, x, c, height, width, channels);
                d_samples[idx] = d_frame[pixBase + c];
            }
            d_ages[slot2 * height * width + y * width + x] = 0;
        }
    }
}


// Launcher wrappers expected by motion_field_cuda.cpp
extern "C" void knnInitKernelLauncher(
    unsigned char* d_samples,
    unsigned short* d_ages,
    int width,
    int height,
    int history,
    int channels,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    knnInitKernel<<<grid, block, 0, stream>>>(
        d_samples,
        d_ages,
        width,
        height,
        history,
        channels
    );
}


extern "C" void knnApplyKernelLauncher(
    const unsigned char* d_frame,
    unsigned char* d_fgmask,
    unsigned char* d_samples,
    unsigned short* d_ages,
    int width,
    int height,
    int history,
    int channels,
    float dist2Threshold,
    int requiredMatches,
    float replaceProb,
    unsigned int frameSeed,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    knnApplyKernel<<<grid, block, 0, stream>>>(
        d_frame,
        d_fgmask,
        d_samples,
        d_ages,
        width,
        height,
        history,
        channels,
        dist2Threshold,
        requiredMatches,
        replaceProb,
        frameSeed
    );
}


// ============================================================
// Region statistics
// ============================================================
extern "C" __global__ void initRegionStatsKernel(
    RegionStatGPU* stats,
    int numLabels,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numLabels) return;

    stats[idx].area = 0;
    stats[idx].minx = width;
    stats[idx].miny = height;
    stats[idx].maxx = -1;
    stats[idx].maxy = -1;
    stats[idx].sumx = 0.0f;
    stats[idx].sumy = 0.0f;
}


extern "C" __global__ void accumulateRegionStatsKernel(
    const int* labels,
    int width,
    int height,
    RegionStatGPU* stats,
    int maxLabels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int label = labels[y * width + x];
    if (label <= 0 || label >= maxLabels) return;

    atomicAdd(&stats[label].area, 1);
    atomicMin(&stats[label].minx, x);
    atomicMin(&stats[label].miny, y);
    atomicMax(&stats[label].maxx, x);
    atomicMax(&stats[label].maxy, y);
    atomicAdd(&stats[label].sumx, (float)x);
    atomicAdd(&stats[label].sumy, (float)y);
}


// ============================================================
// Gaussian vote field
// ============================================================
extern "C" __global__ void gaussianVoteKernel(
    float* votes,       // [H * W * B]
    float* magSum,      // [H * W * B]
    int width,
    int height,
    int numBins,
    const float2* centers,
    const float2* velocities,
    const int* validFlags,
    int numRegions,
    float sigma,
    float minMag
) {
    int rid = blockIdx.x;
    if (rid >= numRegions) return;
    if (validFlags[rid] == 0) return;

    float2 c = centers[rid];
    float2 v = velocities[rid];

    float mag = sqrtf(v.x * v.x + v.y * v.y);
    if (mag < minMag) return;

    int bin = angleToBin(v.x, v.y, numBins);
    if (bin < 0) return;

    int radius = (int)(3.0f * sigma);

    // block is expected to be roughly (2*radius+1, 2*radius+1)
    int lx = threadIdx.x - radius;
    int ly = threadIdx.y - radius;

    int px = (int)roundf(c.x) + lx;
    int py = (int)roundf(c.y) + ly;

    if (px < 0 || px >= width || py < 0 || py >= height) return;

    float d2 = (float)(lx * lx + ly * ly);
    float w = expf(-d2 / (2.0f * sigma * sigma));

    int idx = (py * width + px) * numBins + bin;
    atomicAdd(&votes[idx], w);
    atomicAdd(&magSum[idx], w * mag);
}


// ============================================================
// Dominant field extraction
// ============================================================
extern "C" __global__ void dominantFieldKernel(
    const float* votes,
    const float* magSum,
    int width,
    int height,
    int numBins,
    float minVote,
    int* domBin,
    float* domMag,
    unsigned char* valid
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int base = (y * width + x) * numBins;

    float total = 0.0f;
    float bestVote = -1.0f;
    int bestBin = -1;

    for (int b = 0; b < numBins; ++b) {
        float v = votes[base + b];
        total += v;
        if (v > bestVote) {
            bestVote = v;
            bestBin = b;
        }
    }

    int outIdx = y * width + x;

    if (total < minVote || bestBin < 0) {
        valid[outIdx] = 0;
        domBin[outIdx] = 0;
        domMag[outIdx] = 0.0f;
        return;
    }

    valid[outIdx] = 1;
    domBin[outIdx] = bestBin;
    domMag[outIdx] = (bestVote > 1e-6f) ? (magSum[base + bestBin] / bestVote) : 0.0f;
}

extern "C" void initRegionStatsKernelLauncher(
    RegionStatGPU* stats,
    int numLabels,
    int width,
    int height,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numLabels + threads - 1) / threads;
    initRegionStatsKernel<<<blocks, threads, 0, stream>>>(
        stats, numLabels, width, height
    );
}

extern "C" void accumulateRegionStatsKernelLauncher(
    const int* labels,
    int width,
    int height,
    RegionStatGPU* stats,
    int maxLabels,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    accumulateRegionStatsKernel<<<grid, block, 0, stream>>>(
        labels, width, height, stats, maxLabels
    );
}

extern "C" void gaussianVoteKernelLauncher(
    float* votes,
    float* magSum,
    int width,
    int height,
    int numBins,
    const float2* centers,
    const float2* velocities,
    const int* validFlags,
    int numRegions,
    float sigma,
    float minMag,
    cudaStream_t stream
) {
    dim3 block(21, 21);
    gaussianVoteKernel<<<numRegions, block, 0, stream>>>(
        votes,
        magSum,
        width,
        height,
        numBins,
        centers,
        velocities,
        validFlags,
        numRegions,
        sigma,
        minMag
    );
}

extern "C" void dominantFieldKernelLauncher(
    const float* votes,
    const float* magSum,
    int width,
    int height,
    int numBins,
    float minVote,
    int* domBin,
    float* domMag,
    unsigned char* valid,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    dominantFieldKernel<<<grid, block, 0, stream>>>(
        votes,
        magSum,
        width,
        height,
        numBins,
        minVote,
        domBin,
        domMag,
        valid
    );
}