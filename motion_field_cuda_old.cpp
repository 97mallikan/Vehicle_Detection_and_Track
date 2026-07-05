#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err) \
                      << " at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

struct RegionStatGPU {
    int area;
    int minx, miny, maxx, maxy;
    float sumx, sumy;
};

struct BlobRegion {
    int label = -1;
    int area = 0;
    cv::Rect bbox;
    cv::Point2f center = cv::Point2f(0.f, 0.f);
};

struct TrackState {
    int id = -1;
    float cx = 0.0f;
    float cy = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    int missed = 0;
    cv::Rect bbox;
};

extern "C" {

// -------------------------------
// Custom CUDA KNN
// -------------------------------
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

// -------------------------------
// Region statistics
// -------------------------------
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

void gaussianVoteKernelLauncher(
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
);

void dominantFieldKernelLauncher(
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
);

} // extern "C"


static float dist2(float x1, float y1, float x2, float y2) {
    const float dx = x1 - x2;
    const float dy = y1 - y2;
    return dx * dx + dy * dy;
}


class MotionFieldCUDA {
public:
    MotionFieldCUDA(
        int width,
        int height,
        int bins = 8,
        int knnHistory = 32,
        float knnDist2Threshold = 900.0f,
        int knnRequiredMatches = 2,
        float knnReplaceProb = 0.0020f
    )
        : w(width),
          h(height),
          numBins(bins),
          history(knnHistory),
          dist2Threshold(knnDist2Threshold),
          requiredMatches(knnRequiredMatches),
          replaceProb(knnReplaceProb),
          nextTrackId(1),
          frameSeed(1) {

        if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
            throw std::runtime_error("No CUDA device found.");
        }

        stream = cv::cuda::Stream();

        // GPU buffers
        d_bgr.create(h, w, CV_8UC3);
        d_gray.create(h, w, CV_8UC1);
        d_mask.create(h, w, CV_8UC1);
        d_labels.create(h, w, CV_32S);

        d_votes.create(h, w * numBins, CV_32F);
        d_magSum.create(h, w * numBins, CV_32F);
        d_domBin.create(h, w, CV_32S);
        d_domMag.create(h, w, CV_32F);
        d_valid.create(h, w, CV_8U);

        // CUDA morphology filters
        auto kOpen  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        auto kClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

        morphOpen = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kOpen);
        morphClose = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kClose);

        // Region-stat device memory
        maxLabels = w * h;  // simple upper bound for now
        CUDA_CHECK(cudaMalloc(&d_regionStats, sizeof(RegionStatGPU) * maxLabels));

        // Custom KNN device memory
        // samples: [history][H][W][3]
        const size_t sampleCount = static_cast<size_t>(history) * h * w * 3;
        const size_t ageCount    = static_cast<size_t>(history) * h * w;

        CUDA_CHECK(cudaMalloc(&d_knnSamples, sizeof(unsigned char) * sampleCount));
        CUDA_CHECK(cudaMalloc(&d_knnAges, sizeof(unsigned short) * ageCount));

        d_votes.setTo(cv::Scalar(0), stream);
        d_magSum.setTo(cv::Scalar(0), stream);
        d_domBin.setTo(cv::Scalar(0), stream);
        d_domMag.setTo(cv::Scalar(0), stream);
        d_valid.setTo(cv::Scalar(0), stream);

        // Initialize custom KNN model
        knnInitKernelLauncher(
            d_knnSamples,
            d_knnAges,
            w,
            h,
            history,
            3,
            cv::cuda::StreamAccessor::getStream(stream)
        );
        CUDA_CHECK(cudaGetLastError());
        stream.waitForCompletion();
    }

    ~MotionFieldCUDA() {
        if (d_regionStats) CUDA_CHECK(cudaFree(d_regionStats));
        if (d_knnSamples)  CUDA_CHECK(cudaFree(d_knnSamples));
        if (d_knnAges)     CUDA_CHECK(cudaFree(d_knnAges));
    }

    void process(const cv::Mat& frameBGR, int frameIdx) {
        if (frameBGR.empty()) return;
        if (frameBGR.cols != w || frameBGR.rows != h) {
            throw std::runtime_error("Input frame size does not match MotionFieldCUDA dimensions.");
        }

        uploadFrame(frameBGR);
        applyCustomKNN();
        refineMaskGPU();
        computeConnectedComponents();
        gatherRegionsCPU();
        updateTracksCPU();
        updateGaussianVotesGPU();
        computeDominantFieldGPU();
        drawVisualization(frameBGR);

        frameSeed = static_cast<unsigned int>(frameIdx + 1);
    }

    const cv::Mat& getVis() const {
        return vis;
    }

    const cv::Mat getMaskCPU() {
        cv::Mat out;
        d_mask.download(out, stream);
        stream.waitForCompletion();
        return out;
    }

private:
    int w = 0;
    int h = 0;
    int numBins = 8;

    // KNN params
    int history = 32;
    float dist2Threshold = 900.0f;
    int requiredMatches = 2;
    float replaceProb = 0.0020f;
    unsigned int frameSeed = 1;

    // Track state
    int nextTrackId = 1;
    std::vector<TrackState> tracks;

    // Region state
    int maxLabels = 0;
    RegionStatGPU* d_regionStats = nullptr;
    std::vector<RegionStatGPU> h_stats;
    std::vector<int> activeRegionIds;
    std::vector<float2> regionCenters;
    std::vector<float2> regionVelocities;
    std::vector<int> regionValid;

    // Custom KNN device buffers
    unsigned char* d_knnSamples = nullptr;
    unsigned short* d_knnAges = nullptr;

    // CUDA/OpenCV buffers
    cv::cuda::Stream stream;
    cv::cuda::GpuMat d_bgr;
    cv::cuda::GpuMat d_gray;
    cv::cuda::GpuMat d_mask;
    cv::cuda::GpuMat d_labels;

    cv::cuda::GpuMat d_votes;
    cv::cuda::GpuMat d_magSum;
    cv::cuda::GpuMat d_domBin;
    cv::cuda::GpuMat d_domMag;
    cv::cuda::GpuMat d_valid;

    cv::Ptr<cv::cuda::Filter> morphOpen;
    cv::Ptr<cv::cuda::Filter> morphClose;

    cv::Mat vis;

private:
    void uploadFrame(const cv::Mat& frameBGR) {
        d_bgr.upload(frameBGR, stream);
        cv::cuda::cvtColor(d_bgr, d_gray, cv::COLOR_BGR2GRAY, 0, stream);
    }

    void applyCustomKNN() {
        knnApplyKernelLauncher(
            d_bgr.ptr<unsigned char>(),
            d_mask.ptr<unsigned char>(),
            d_knnSamples,
            d_knnAges,
            w,
            h,
            history,
            3,
            dist2Threshold,
            requiredMatches,
            replaceProb,
            frameSeed,
            cv::cuda::StreamAccessor::getStream(stream)
        );
        CUDA_CHECK(cudaGetLastError());
    }

    void refineMaskGPU() {
        cv::cuda::threshold(d_mask, d_mask, 200.0, 255.0, cv::THRESH_BINARY, stream);
        morphOpen->apply(d_mask, d_mask, stream);
        morphClose->apply(d_mask, d_mask, stream);
    }

    void computeConnectedComponents() {
        cv::cuda::connectedComponents(d_mask, d_labels, 8, CV_32S);
        stream.waitForCompletion();
    }

    void gatherRegionsCPU() {
        cudaStream_t rawStream = cv::cuda::StreamAccessor::getStream(stream);

        initRegionStatsKernelLauncher(
            d_regionStats,
            maxLabels,
            w,
            h,
            rawStream
        );
        CUDA_CHECK(cudaGetLastError());

        accumulateRegionStatsKernelLauncher(
            d_labels.ptr<int>(),
            w,
            h,
            d_regionStats,
            maxLabels,
            rawStream
        );
        CUDA_CHECK(cudaGetLastError());

        h_stats.resize(maxLabels);
        CUDA_CHECK(cudaMemcpy(
            h_stats.data(),
            d_regionStats,
            sizeof(RegionStatGPU) * maxLabels,
            cudaMemcpyDeviceToHost
        ));

        activeRegionIds.clear();
        regionCenters.clear();

        for (int label = 1; label < maxLabels; ++label) {
            const auto& s = h_stats[label];

            if (s.area < 700) continue;
            if (s.maxx < s.minx || s.maxy < s.miny) continue;

            activeRegionIds.push_back(label);

            float2 c;
            c.x = s.sumx / std::max(1, s.area);
            c.y = s.sumy / std::max(1, s.area);
            regionCenters.push_back(c);
        }
    }

    void updateTracksCPU() {
        // Predict existing tracks
        for (auto& t : tracks) {
            t.cx += t.vx;
            t.cy += t.vy;
            t.missed++;
        }

        std::vector<bool> matchedTrack(tracks.size(), false);

        regionVelocities.clear();
        regionValid.clear();

        for (size_t i = 0; i < regionCenters.size(); ++i) {
            const auto c = regionCenters[i];

            int best = -1;
            float bestD = 70.0f * 70.0f;

            for (size_t j = 0; j < tracks.size(); ++j) {
                const float d = dist2(c.x, c.y, tracks[j].cx, tracks[j].cy);
                if (d < bestD) {
                    bestD = d;
                    best = static_cast<int>(j);
                }
            }

            if (best >= 0) {
                auto& t = tracks[best];

                const float newVx = c.x - t.cx;
                const float newVy = c.y - t.cy;

                t.vx = 0.8f * t.vx + 0.2f * newVx;
                t.vy = 0.8f * t.vy + 0.2f * newVy;
                t.cx = c.x;
                t.cy = c.y;
                t.missed = 0;
                matchedTrack[best] = true;

                const auto& s = h_stats[activeRegionIds[i]];
                t.bbox = cv::Rect(
                    s.minx,
                    s.miny,
                    s.maxx - s.minx + 1,
                    s.maxy - s.miny + 1
                );

                regionVelocities.push_back(float2{t.vx, t.vy});
                regionValid.push_back(1);
            } else {
                TrackState t;
                t.id = nextTrackId++;
                t.cx = c.x;
                t.cy = c.y;
                t.vx = 0.0f;
                t.vy = 0.0f;
                t.missed = 0;

                const auto& s = h_stats[activeRegionIds[i]];
                t.bbox = cv::Rect(
                    s.minx,
                    s.miny,
                    s.maxx - s.minx + 1,
                    s.maxy - s.miny + 1
                );

                tracks.push_back(t);

                regionVelocities.push_back(float2{0.0f, 0.0f});
                regionValid.push_back(1);
            }
        }

        std::vector<TrackState> kept;
        kept.reserve(tracks.size());
        for (auto& t : tracks) {
            if (t.missed <= 8) kept.push_back(t);
        }
        tracks.swap(kept);
    }

    void updateGaussianVotesGPU() {
        d_votes.setTo(cv::Scalar(0), stream);
        d_magSum.setTo(cv::Scalar(0), stream);

        if (regionCenters.empty()) return;

        float2* d_centers = nullptr;
        float2* d_vels = nullptr;
        int* d_flags = nullptr;

        const int n = static_cast<int>(regionCenters.size());

        CUDA_CHECK(cudaMalloc(&d_centers, sizeof(float2) * n));
        CUDA_CHECK(cudaMalloc(&d_vels, sizeof(float2) * n));
        CUDA_CHECK(cudaMalloc(&d_flags, sizeof(int) * n));

        CUDA_CHECK(cudaMemcpy(
            d_centers,
            regionCenters.data(),
            sizeof(float2) * n,
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_vels,
            regionVelocities.data(),
            sizeof(float2) * n,
            cudaMemcpyHostToDevice
        ));
        CUDA_CHECK(cudaMemcpy(
            d_flags,
            regionValid.data(),
            sizeof(int) * n,
            cudaMemcpyHostToDevice
        ));

        cudaStream_t rawStream = cv::cuda::StreamAccessor::getStream(stream);

        gaussianVoteKernelLauncher(
            d_votes.ptr<float>(),
            d_magSum.ptr<float>(),
            w,
            h,
            numBins,
            d_centers,
            d_vels,
            d_flags,
            n,
            8.0f,
            0.15f,
            rawStream
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_centers));
        CUDA_CHECK(cudaFree(d_vels));
        CUDA_CHECK(cudaFree(d_flags));
    }

    void computeDominantFieldGPU() {
        cudaStream_t rawStream = cv::cuda::StreamAccessor::getStream(stream);

        dominantFieldKernelLauncher(
            d_votes.ptr<float>(),
            d_magSum.ptr<float>(),
            w,
            h,
            numBins,
            5.0f,
            d_domBin.ptr<int>(),
            d_domMag.ptr<float>(),
            d_valid.ptr<unsigned char>(),
            rawStream
        );
        CUDA_CHECK(cudaGetLastError());
    }

    void drawVisualization(const cv::Mat& frameBGR) {
        vis = frameBGR.clone();

        cv::Mat domBin, domMag, valid;
        d_domBin.download(domBin, stream);
        d_domMag.download(domMag, stream);
        d_valid.download(valid, stream);
        stream.waitForCompletion();

        const int step = 12;
        const float scale = 3.0f;

        for (int y = 0; y < h; y += step) {
            for (int x = 0; x < w; x += step) {
                if (valid.at<unsigned char>(y, x) == 0) continue;

                const int b = domBin.at<int>(y, x);
                const float mag = domMag.at<float>(y, x);
                if (mag < 0.15f) continue;

                const float angle = (360.0f / numBins) * b;
                const float rad = angle * 3.14159265358979323846f / 180.0f;

                const int dx = static_cast<int>(std::round(std::cos(rad) * mag * scale));
                const int dy = static_cast<int>(std::round(std::sin(rad) * mag * scale));

                cv::arrowedLine(
                    vis,
                    cv::Point(x, y),
                    cv::Point(x + dx, y + dy),
                    cv::Scalar(0, 255, 0),
                    1,
                    cv::LINE_AA,
                    0,
                    0.3
                );
            }
        }

        for (const auto& t : tracks) {
            cv::rectangle(vis, t.bbox, cv::Scalar(255, 0, 0), 2);

            cv::putText(
                vis,
                "ID:" + std::to_string(t.id) +
                " vx:" + cv::format("%.2f", t.vx) +
                " vy:" + cv::format("%.2f", t.vy),
                cv::Point(t.bbox.x, std::max(20, t.bbox.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 255, 255),
                1,
                cv::LINE_AA
            );
        }
    }
};


int main() {
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
        std::cerr << "No CUDA-enabled OpenCV device found" << std::endl;
        return -1;
    }

    cv::VideoCapture cap("/home/anurag/python-environments/yolov8-object-tracking/RealLifeVideo.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Cannot open input file" << std::endl;
        return -1;
    }

    cv::Mat first;
    if (!cap.read(first)) {
        std::cerr << "Cannot read first frame" << std::endl;
        return -1;
    }

    MotionFieldCUDA engine(first.cols, first.rows, 8);
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    cv::Mat frame;
    int frameIdx = 0;

    while (cap.read(frame)) {
        frameIdx++;
        engine.process(frame, frameIdx);

        cv::imshow("CUDA Motion Field", engine.getVis());
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break;
    }

    return 0;
}