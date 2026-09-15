#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaoptflow.hpp>

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(_err), __LINE__); \
            return 0; \
        } \
    } while (0)

// Public C API structures.
// These must remain outside the anonymous namespace so the exported
// extern "C" functions that use them retain global symbol visibility.

struct BlobRegion {
    int label;
    int area;
    int x;
    int y;
    int w;
    int h;
    float cx;
    float cy;
};

struct TrackCenter {
    int x;
    int y;
};

struct ObjectFlowResultHost {
    float majority_vx;
    float majority_vy;
    float majority_mag;
    int valid_count;
    int anomaly_count;
    float anomaly_ratio;
    int kernel_x1;
    int kernel_y1;
    int kernel_x2;
    int kernel_y2;
};

struct FlowPointHost {
    int x0;
    int y0;
    float vx;
    float vy;
    float mag;
    int anomalous;
};

namespace {

constexpr int KSIZE = 32;
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int THREADS_PER_BLOCK = BLOCK_X * BLOCK_Y;
constexpr int DEFAULT_POINT_STRIDE = 128;

__constant__ float c_gauss32[KSIZE * KSIZE];

struct RegionStatGPU {
    int area;
    int minx, miny, maxx, maxy;
    float sumx, sumy;
};

struct FlowPointGpu {
    short lx;
    short ly;
    float vx;
    float vy;
    float mag;
    unsigned char anomalous;
    unsigned char valid;
    unsigned char _pad0;
    unsigned char _pad1;
};

struct ObjectFlowResultGpu {
    float majority_vx;
    float majority_vy;
    float majority_mag;
    int valid_count;
    int anomaly_count;
    float anomaly_ratio;
    int kernel_x1;
    int kernel_y1;
    int kernel_x2;
    int kernel_y2;
};

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

__device__ __forceinline__ float fast_mag(float x, float y) {
    return sqrtf(x * x + y * y);
}

std::vector<float> createGaussianMap32(float sigma = 8.0f) {
    std::vector<float> g(KSIZE * KSIZE, 0.0f);
    const float center = (KSIZE - 1) * 0.5f;
    float sum = 0.0f;
    for (int y = 0; y < KSIZE; ++y) {
        for (int x = 0; x < KSIZE; ++x) {
            const float dx = x - center;
            const float dy = y - center;
            const float w = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            g[y * KSIZE + x] = w;
            sum += w;
        }
    }
    if (sum > 1e-12f) {
        for (float& v : g) v /= sum;
    }
    return g;
}

bool uploadGaussianMapOnce() {
    static bool uploaded = false;
    if (uploaded) return true;
    const auto g = createGaussianMap32();
    cudaError_t err = cudaMemcpyToSymbol(c_gauss32, g.data(), sizeof(float) * KSIZE * KSIZE);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpyToSymbol(c_gauss32) failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    uploaded = true;
    return true;
}

__global__ void analyze_object_flow_kernel(
    const float* __restrict__ flow_x,
    const float* __restrict__ flow_y,
    const TrackCenter* __restrict__ centers,
    int num_objects,
    int width,
    int height,
    float min_mag,
    float cos_angle_threshold,
    float mag_ratio_low,
    ObjectFlowResultGpu* __restrict__ results,
    FlowPointGpu* __restrict__ point_out,
    int point_stride
) {
    __shared__ float s_gauss[KSIZE * KSIZE];
    __shared__ float s_sumx[THREADS_PER_BLOCK];
    __shared__ float s_sumy[THREADS_PER_BLOCK];
    __shared__ float s_sumw[THREADS_PER_BLOCK];
    __shared__ int s_valid[THREADS_PER_BLOCK];
    __shared__ int s_anom[THREADS_PER_BLOCK];
    __shared__ int s_counter;
    __shared__ float s_mvx;
    __shared__ float s_mvy;
    __shared__ float s_mmag;
    __shared__ int s_out_count;

    const int obj_idx = blockIdx.x;
    if (obj_idx >= num_objects) return;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int i = tid; i < KSIZE * KSIZE; i += THREADS_PER_BLOCK) {
        s_gauss[i] = c_gauss32[i];
    }
    if (tid == 0) {
        s_counter = 0;
        s_out_count = 0;
    }
    __syncthreads();

    const TrackCenter c = centers[obj_idx];
    const int x0 = c.x - KSIZE / 2;
    const int y0 = c.y - KSIZE / 2;

    float local_sumx = 0.0f;
    float local_sumy = 0.0f;
    float local_sumw = 0.0f;
    int local_valid = 0;

    for (int ly = threadIdx.y; ly < KSIZE; ly += blockDim.y) {
        for (int lx = threadIdx.x; lx < KSIZE; lx += blockDim.x) {
            const int gx = clampi(x0 + lx, 0, width - 1);
            const int gy = clampi(y0 + ly, 0, height - 1);
            const int idx = gy * width + gx;

            const float vx = flow_x[idx];
            const float vy = flow_y[idx];
            const float mag = fast_mag(vx, vy);
            if (mag >= min_mag) {
                const float w = s_gauss[ly * KSIZE + lx];
                local_sumx += vx * w;
                local_sumy += vy * w;
                local_sumw += w;
                local_valid += 1;
            }
        }
    }

    s_sumx[tid] = local_sumx;
    s_sumy[tid] = local_sumy;
    s_sumw[tid] = local_sumw;
    s_valid[tid] = local_valid;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sumx[tid] += s_sumx[tid + stride];
            s_sumy[tid] += s_sumy[tid + stride];
            s_sumw[tid] += s_sumw[tid + stride];
            s_valid[tid] += s_valid[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mvx = 0.0f, mvy = 0.0f, mmag = 0.0f;
        if (s_sumw[0] > 1e-12f) {
            mvx = s_sumx[0] / s_sumw[0];
            mvy = s_sumy[0] / s_sumw[0];
            mmag = fast_mag(mvx, mvy);
        }
        s_mvx = mvx;
        s_mvy = mvy;
        s_mmag = mmag;
    }
    __syncthreads();

    int local_anom = 0;
    for (int ly = threadIdx.y; ly < KSIZE; ly += blockDim.y) {
        for (int lx = threadIdx.x; lx < KSIZE; lx += blockDim.x) {
            const int gx = clampi(x0 + lx, 0, width - 1);
            const int gy = clampi(y0 + ly, 0, height - 1);
            const int idx = gy * width + gx;

            const float vx = flow_x[idx];
            const float vy = flow_y[idx];
            const float mag = fast_mag(vx, vy);
            if (mag < min_mag || s_mmag < 1e-6f) continue;

            float dot = (vx * s_mvx + vy * s_mvy) / (mag * s_mmag + 1e-6f);
            dot = fminf(1.0f, fmaxf(-1.0f, dot));
            const float mag_ratio = mag / (s_mmag + 1e-6f);
            const bool is_anom = (dot < cos_angle_threshold) || (mag_ratio < mag_ratio_low);
            if (is_anom) local_anom++;

            if (point_out && point_stride > 0) {
                int slot = atomicAdd(&s_counter, 1);
                if (slot < point_stride) {
                    FlowPointGpu p{};
                    p.lx = static_cast<short>(lx);
                    p.ly = static_cast<short>(ly);
                    p.vx = vx;
                    p.vy = vy;
                    p.mag = mag;
                    p.anomalous = is_anom ? 1 : 0;
                    p.valid = 1;
                    point_out[obj_idx * point_stride + slot] = p;
                }
            }
        }
    }

    s_anom[tid] = local_anom;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_anom[tid] += s_anom[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        ObjectFlowResultGpu out{};
        out.majority_vx = s_mvx;
        out.majority_vy = s_mvy;
        out.majority_mag = s_mmag;
        out.valid_count = s_valid[0];
        out.anomaly_count = s_anom[0];
        out.anomaly_ratio = (s_valid[0] > 0) ? (static_cast<float>(s_anom[0]) / static_cast<float>(s_valid[0])) : 0.0f;
        out.kernel_x1 = x0;
        out.kernel_y1 = y0;
        out.kernel_x2 = x0 + KSIZE;
        out.kernel_y2 = y0 + KSIZE;
        results[obj_idx] = out;

        if (point_out && point_stride > 0) {
            s_out_count = min(s_counter, point_stride);
            for (int i = s_out_count; i < point_stride; ++i) {
                FlowPointGpu z{};
                point_out[obj_idx * point_stride + i] = z;
            }
        }
    }
}

} // namespace

extern "C" {

// Launchers provided by motion_kernels.cu
void knnInitKernelLauncher(
    unsigned char* d_samples,
    unsigned short* d_ages,
    int width,
    int height,
    int history,
    int channels,
    cudaStream_t stream
);

void knnApplyKernelLauncher(
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
);

void initRegionStatsKernelLauncher(
    RegionStatGPU* stats,
    int numLabels,
    int width,
    int height,
    cudaStream_t stream
);

void accumulateRegionStatsKernelLauncher(
    const int* labels,
    int width,
    int height,
    RegionStatGPU* stats,
    int maxLabels,
    cudaStream_t stream
);

}

class MotionFieldCUDA {
public:
    MotionFieldCUDA(int width, int height, int max_regions = 2048)
        : width_(width), height_(height), max_regions_(max_regions) {
        if (!uploadGaussianMapOnce()) {
            throw std::runtime_error("Failed to upload Gaussian map");
        }
        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
            throw std::runtime_error("No CUDA-enabled device found");
        }

        stream_ = cv::cuda::Stream();

        d_bgr_.create(height_, width_, CV_8UC3);
        d_gray_.create(height_, width_, CV_8UC1);
        d_prev_gray_.create(height_, width_, CV_8UC1);
        d_fgmask_.create(height_, width_, CV_8UC1);
        d_labels_.create(height_, width_, CV_32S);
        d_flow_xy_.create(height_, width_, CV_32FC2);
        d_flow_x_.create(height_, width_, CV_32F);
        d_flow_y_.create(height_, width_, CV_32F);

        auto kOpen = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        auto kClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        morph_open_ = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kOpen);
        morph_close_ = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kClose);

        farneback_ = cv::cuda::FarnebackOpticalFlow::create(5, 0.5, false, 13, 10, 5, 1.1, 0);

        max_labels_ = width_ * height_;
        cudaMalloc(&d_region_stats_, sizeof(RegionStatGPU) * max_labels_);

        const size_t sampleCount = static_cast<size_t>(history_) * height_ * width_ * 3;
        const size_t ageCount = static_cast<size_t>(history_) * height_ * width_;
        cudaMalloc(&d_knn_samples_, sizeof(unsigned char) * sampleCount);
        cudaMalloc(&d_knn_ages_, sizeof(unsigned short) * ageCount);

        knnInitKernelLauncher(
            d_knn_samples_, d_knn_ages_, width_, height_, history_, 3,
            cv::cuda::StreamAccessor::getStream(stream_)
        );
        cudaStreamSynchronize(cv::cuda::StreamAccessor::getStream(stream_));
    }

    ~MotionFieldCUDA() {
        if (d_region_stats_) cudaFree(d_region_stats_);
        if (d_knn_samples_) cudaFree(d_knn_samples_);
        if (d_knn_ages_) cudaFree(d_knn_ages_);
    }

    bool process(const unsigned char* frame_bgr, int step, float /*learning_rate*/, int min_area) {
        if (!frame_bgr) return false;

        cv::Mat wrapped(height_, width_, CV_8UC3, const_cast<unsigned char*>(frame_bgr), step);
        d_bgr_.upload(wrapped, stream_);
        cv::cuda::cvtColor(d_bgr_, d_gray_, cv::COLOR_BGR2GRAY, 0, stream_);

        knnApplyKernelLauncher(
            d_bgr_.ptr<unsigned char>(),
            d_fgmask_.ptr<unsigned char>(),
            d_knn_samples_,
            d_knn_ages_,
            width_, height_, history_, 3,
            dist2Threshold_, requiredMatches_, replaceProb_, frameSeed_,
            cv::cuda::StreamAccessor::getStream(stream_)
        );

        cv::cuda::threshold(d_fgmask_, d_fgmask_, 200.0, 255.0, cv::THRESH_BINARY, stream_);
        morph_open_->apply(d_fgmask_, d_fgmask_, stream_);
        morph_close_->apply(d_fgmask_, d_fgmask_, stream_);
        cv::cuda::connectedComponents(d_fgmask_, d_labels_, 8, CV_32S);
        stream_.waitForCompletion();

        gatherRegions(min_area);

        if (has_prev_gray_) {
            farneback_->calc(d_prev_gray_, d_gray_, d_flow_xy_, stream_);
            std::vector<cv::cuda::GpuMat> chans;
            cv::cuda::split(d_flow_xy_, chans, stream_);
            if (chans.size() >= 2) {
                d_flow_x_ = chans[0];
                d_flow_y_ = chans[1];
            }
        } else {
            d_flow_x_.setTo(cv::Scalar(0), stream_);
            d_flow_y_.setTo(cv::Scalar(0), stream_);
            has_prev_gray_ = true;
        }

        d_gray_.copyTo(d_prev_gray_, stream_);
        stream_.waitForCompletion();
        frameSeed_ += 1;
        return true;
    }

    void downloadMask(unsigned char* dst) {
        if (!dst) return;
        cv::Mat out(height_, width_, CV_8UC1, dst);
        d_fgmask_.download(out, stream_);
        stream_.waitForCompletion();
    }

    int blobCount() const { return static_cast<int>(blobs_.size()); }

    int getBlobs(BlobRegion* out, int max_out) const {
        if (!out || max_out <= 0) return 0;
        int n = std::min<int>(max_out, blobs_.size());
        std::memcpy(out, blobs_.data(), sizeof(BlobRegion) * n);
        return n;
    }

    int analyzeObjects(
        const TrackCenter* centers,
        int num_objects,
        float min_mag,
        float angle_thr_deg,
        float mag_ratio_low,
        ObjectFlowResultHost* out_results,
        FlowPointHost* out_points,
        int point_stride
    ) {
        if (!has_prev_gray_ || num_objects <= 0 || !centers || !out_results) return 0;
        if (point_stride <= 0) point_stride = DEFAULT_POINT_STRIDE;

        cudaStream_t raw = cv::cuda::StreamAccessor::getStream(stream_);
        TrackCenter* d_centers = nullptr;
        ObjectFlowResultGpu* d_results = nullptr;
        FlowPointGpu* d_points = nullptr;

        cudaMalloc(&d_centers, sizeof(TrackCenter) * num_objects);
        cudaMalloc(&d_results, sizeof(ObjectFlowResultGpu) * num_objects);
        cudaMemcpyAsync(d_centers, centers, sizeof(TrackCenter) * num_objects, cudaMemcpyHostToDevice, raw);

        if (out_points) {
            cudaMalloc(&d_points, sizeof(FlowPointGpu) * num_objects * point_stride);
            cudaMemsetAsync(d_points, 0, sizeof(FlowPointGpu) * num_objects * point_stride, raw);
        }

        const float cos_thr = std::cos(angle_thr_deg * 3.1415926535f / 180.0f);
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid(num_objects);

        analyze_object_flow_kernel<<<grid, block, 0, raw>>>(
            d_flow_x_.ptr<float>(), d_flow_y_.ptr<float>(), d_centers, num_objects,
            width_, height_, min_mag, cos_thr, mag_ratio_low,
            d_results, d_points, point_stride
        );

        std::vector<ObjectFlowResultGpu> h_results(num_objects);
        cudaMemcpyAsync(h_results.data(), d_results, sizeof(ObjectFlowResultGpu) * num_objects, cudaMemcpyDeviceToHost, raw);

        std::vector<FlowPointGpu> h_points;
        if (out_points && d_points) {
            h_points.resize(num_objects * point_stride);
            cudaMemcpyAsync(h_points.data(), d_points, sizeof(FlowPointGpu) * num_objects * point_stride, cudaMemcpyDeviceToHost, raw);
        }
        cudaStreamSynchronize(raw);

        for (int i = 0; i < num_objects; ++i) {
            out_results[i].majority_vx = h_results[i].majority_vx;
            out_results[i].majority_vy = h_results[i].majority_vy;
            out_results[i].majority_mag = h_results[i].majority_mag;
            out_results[i].valid_count = h_results[i].valid_count;
            out_results[i].anomaly_count = h_results[i].anomaly_count;
            out_results[i].anomaly_ratio = h_results[i].anomaly_ratio;
            out_results[i].kernel_x1 = h_results[i].kernel_x1;
            out_results[i].kernel_y1 = h_results[i].kernel_y1;
            out_results[i].kernel_x2 = h_results[i].kernel_x2;
            out_results[i].kernel_y2 = h_results[i].kernel_y2;
        }

        if (out_points) {
            for (int i = 0; i < num_objects; ++i) {
                for (int j = 0; j < point_stride; ++j) {
                    const auto& p = h_points[i * point_stride + j];
                    auto& q = out_points[i * point_stride + j];
                    if (!p.valid || p.mag <= 0.0f) {
                        q = FlowPointHost{};
                        continue;
                    }
                    q.x0 = h_results[i].kernel_x1 + static_cast<int>(p.lx);
                    q.y0 = h_results[i].kernel_y1 + static_cast<int>(p.ly);
                    q.vx = p.vx;
                    q.vy = p.vy;
                    q.mag = p.mag;
                    q.anomalous = p.anomalous ? 1 : 0;
                }
            }
        }

        if (d_points) cudaFree(d_points);
        cudaFree(d_results);
        cudaFree(d_centers);
        return num_objects;
    }

private:
    void gatherRegions(int min_area) {
        cudaStream_t raw = cv::cuda::StreamAccessor::getStream(stream_);
        initRegionStatsKernelLauncher(d_region_stats_, max_labels_, width_, height_, raw);
        accumulateRegionStatsKernelLauncher(d_labels_.ptr<int>(), width_, height_, d_region_stats_, max_labels_, raw);

        host_stats_.resize(max_labels_);
        cudaMemcpyAsync(host_stats_.data(), d_region_stats_, sizeof(RegionStatGPU) * max_labels_, cudaMemcpyDeviceToHost, raw);
        cudaStreamSynchronize(raw);

        blobs_.clear();
        blobs_.reserve(max_regions_);
        for (int label = 1; label < max_labels_ && static_cast<int>(blobs_.size()) < max_regions_; ++label) {
            const auto& s = host_stats_[label];
            if (s.area < min_area) continue;
            if (s.maxx < s.minx || s.maxy < s.miny) continue;

            BlobRegion b{};
            b.label = label;
            b.area = s.area;
            b.x = s.minx;
            b.y = s.miny;
            b.w = s.maxx - s.minx + 1;
            b.h = s.maxy - s.miny + 1;
            b.cx = s.sumx / std::max(1, s.area);
            b.cy = s.sumy / std::max(1, s.area);
            blobs_.push_back(b);
        }
    }

private:
    int width_ = 0;
    int height_ = 0;
    int max_regions_ = 2048;
    int history_ = 32;
    float dist2Threshold_ = 900.0f;
    int requiredMatches_ = 2;
    float replaceProb_ = 0.0020f;
    unsigned int frameSeed_ = 1;
    bool has_prev_gray_ = false;
    int max_labels_ = 0;

    cv::cuda::Stream stream_;
    cv::cuda::GpuMat d_bgr_;
    cv::cuda::GpuMat d_gray_;
    cv::cuda::GpuMat d_prev_gray_;
    cv::cuda::GpuMat d_fgmask_;
    cv::cuda::GpuMat d_labels_;
    cv::cuda::GpuMat d_flow_xy_;
    cv::cuda::GpuMat d_flow_x_;
    cv::cuda::GpuMat d_flow_y_;

    cv::Ptr<cv::cuda::Filter> morph_open_;
    cv::Ptr<cv::cuda::Filter> morph_close_;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback_;

    RegionStatGPU* d_region_stats_ = nullptr;
    unsigned char* d_knn_samples_ = nullptr;
    unsigned short* d_knn_ages_ = nullptr;

    std::vector<RegionStatGPU> host_stats_;
    std::vector<BlobRegion> blobs_;
};

extern "C" {

void* motion_create(int width, int height, int max_regions) {
    try {
        return reinterpret_cast<void*>(new MotionFieldCUDA(width, height, max_regions));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "motion_create failed: %s\n", e.what());
        return nullptr;
    }
}

void motion_destroy(void* handle) {
    if (!handle) return;
    delete reinterpret_cast<MotionFieldCUDA*>(handle);
}

int motion_process(void* handle, unsigned char* frame_bgr, int step, float learning_rate, int min_area) {
    if (!handle || !frame_bgr) return 0;
    MotionFieldCUDA* self = reinterpret_cast<MotionFieldCUDA*>(handle);
    return self->process(frame_bgr, step, learning_rate, min_area) ? 1 : 0;
}

void motion_download_mask(void* handle, unsigned char* dst_mask) {
    if (!handle || !dst_mask) return;
    reinterpret_cast<MotionFieldCUDA*>(handle)->downloadMask(dst_mask);
}

int motion_get_blob_count(void* handle) {
    if (!handle) return 0;
    return reinterpret_cast<MotionFieldCUDA*>(handle)->blobCount();
}

int motion_get_blobs(void* handle, BlobRegion* out_blobs, int max_out) {
    if (!handle || !out_blobs || max_out <= 0) return 0;
    return reinterpret_cast<MotionFieldCUDA*>(handle)->getBlobs(out_blobs, max_out);
}

int motion_analyze_objects(
    void* handle,
    const TrackCenter* centers,
    int num_objects,
    float min_mag,
    float angle_thr_deg,
    float mag_ratio_low,
    ObjectFlowResultHost* out_results,
    FlowPointHost* out_points,
    int point_stride
) {
    if (!handle || !centers || !out_results || num_objects <= 0) return 0;
    return reinterpret_cast<MotionFieldCUDA*>(handle)->analyzeObjects(
        centers, num_objects, min_mag, angle_thr_deg, mag_ratio_low,
        out_results, out_points, point_stride
    );
}

} // extern "C"
