#ifndef _MAIN_H_
#define _MAIN_H_
// Includes Opencv
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
// Includes STD libs
#include <vector>
#include <cstddef>
#include <utility>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <cstdarg>
#include <random>
#include <unordered_map>
#include <stdexcept>
// Includes Boost filesystem
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

// Define some const var
#define MAX_IMAGES 32
#define NEIGHBOUR_NUM 9
#define MAX_SEARCH_RADIUS 4096
#define DEBUG_POINT_X 753
#define DEBUG_POINT_Y 259
//#define DEBUG_COST_LINE
//#define DEBUG_NEIGHBOUR

using namespace boost::filesystem;

struct Camera {
	float K[9];
	float R[9];
	float t[3];
	float c[3];
	int height;
	int width;
	float depth_min;
	float depth_max;
};

struct PointList {
	float3 coord;
	float3 normal;
	float3 color;
};

enum RunState {
	FIRST_INIT,
	REFINE_INIT,
	REFINE_ITER,
};

enum PixelState {
	WEAK,
	STRONG,
	UNKNOWN
};

struct PatchMatchParams {
	int max_iterations = 3;
	int num_images = 5;
	float sigma_spatial = 5.0f;
	float sigma_color = 3.0f;
	int top_k = 4;
	float depth_min = 0.0f;
	float depth_max = 1.0f;
	bool geom_consistency = false;
	int strong_radius = 5;
	int strong_increment = 2;
	int weak_radius = 5;
	int weak_increment = 5;
	bool use_APD = true;
	int weak_peak_radius = 2;
	int rotate_time = 4;
	float ransac_threshold = 0.005;
	float geom_factor = 0.2f;
	RunState state;
};

struct Problem {
	int index;
	int ref_image_id;
	std::vector<int> src_image_ids;
	path dense_folder;
	path result_folder;
	int initial_image_scale = 4;
	int scale_size = 1;
	std::shared_ptr<class ScaledImageCache> scaled_image_cache;
	PatchMatchParams params;
	bool show_medium_result = false;
	int iteration;
};

// Holds the already resized grayscale views needed by the next APD problem.
// The per-scale files live on disk; only the active reference/source set is
// retained in memory between successive ProcessProblem calls.
class ScaledImageCache {
public:
	explicit ScaledImageCache(size_t capacity);
	void Retain(const std::vector<int> &image_ids);
	void SetOriginalSize(int image_id, const cv::Size &size);
	cv::Size OriginalSize(int image_id) const;
	cv::Mat Get(int image_id, const boost::filesystem::path &scaled_image_path);
	void Clear();
	void ResetStats();
	size_t CacheHits() const { return cache_hits; }
	size_t CacheMisses() const { return cache_misses; }
	double DiskReadSeconds() const { return disk_read_seconds; }
	size_t Capacity() const { return capacity; }

private:
	size_t capacity;
	size_t cache_hits = 0;
	size_t cache_misses = 0;
	double disk_read_seconds = 0.0;
	std::unordered_map<int, cv::Mat> images;
	std::unordered_map<int, cv::Size> original_sizes;
};

boost::filesystem::path ScaledImagePath(const boost::filesystem::path &dense_folder,
	int image_id, int initial_image_scale, int pyramid_scale);

#endif // !_MAIN_H_
