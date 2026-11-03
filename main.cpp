#include "main.h"
#include "APD.h"

using namespace boost::filesystem;

void GenerateSampleList(const path &dense_folder, std::vector<Problem> &problems)
{
	path cluster_list_path = dense_folder / path("pair.txt");
	problems.clear();
	ifstream file(cluster_list_path);
	std::stringstream iss;
	std::string line;

	int num_images;
	iss.clear();
	std::getline(file, line);
	iss.str(line);
	iss >> num_images;

	for (int i = 0; i < num_images; ++i) {
		Problem problem;
		problem.index = i;
		problem.src_image_ids.clear();
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> problem.ref_image_id;

		problem.dense_folder = dense_folder;
		problem.result_folder = dense_folder / path("APD") / path(ToFormatIndex(problem.ref_image_id));
		create_directory(problem.result_folder);

		int num_src_images;
		iss.clear();
		std::getline(file, line);
		iss.str(line);
		iss >> num_src_images;
		for (int j = 0; j < num_src_images; ++j) {
			int id;
			float score;
			iss >> id >> score;
			if (score <= 0.0f) {
				continue;
			}
			problem.src_image_ids.push_back(id);
		}
		problems.push_back(problem);
	}
}

std::vector<path> PrepareScaleImages(const std::vector<Problem> &problems,
	int pyramid_scale, ScaledImageCache &cache) {
	std::unordered_set<int> image_ids;
	for (const auto &problem : problems) {
		image_ids.insert(problem.ref_image_id);
		image_ids.insert(problem.src_image_ids.begin(), problem.src_image_ids.end());
	}
	std::vector<path> files;
	files.reserve(image_ids.size());
	const auto started = std::chrono::steady_clock::now();
	const int initial_scale = problems.front().initial_image_scale;
	const int effective_scale = initial_scale * pyramid_scale;
	const path cache_dir = problems.front().dense_folder / path("APD") / path("scaled_images");
	create_directories(cache_dir);
	size_t completed = 0;
	for (const int image_id : image_ids) {
		const path image_path = problems.front().dense_folder / path("images") /
			path(ToFormatIndex(image_id) + ".jpg");
		cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
		if (image.empty()) throw std::runtime_error("Cannot read image: " + image_path.string());
		const cv::Size original_size = image.size();
		cache.SetOriginalSize(image_id, original_size);
		const int width = std::max(1, static_cast<int>(std::round(
			original_size.width / static_cast<double>(effective_scale))));
		const int height = std::max(1, static_cast<int>(std::round(
			original_size.height / static_cast<double>(effective_scale))));
		cv::Mat image_float, scaled;
		image.convertTo(image_float, CV_32FC1);
		if (width == image.cols && height == image.rows) scaled = image_float;
		else cv::resize(image_float, scaled, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
		const path scaled_path = ScaledImagePath(problems.front().dense_folder, image_id,
			initial_scale, pyramid_scale);
		if (exists(scaled_path)) remove(scaled_path);
		if (!WriteBinMat(scaled_path, scaled)) {
			throw std::runtime_error("Cannot write prepared scaled image: " + scaled_path.string());
		}
		files.push_back(scaled_path);
		if (++completed % 20 == 0 || completed == image_ids.size()) {
			std::cout << "Prepared scaled images " << completed << "/" << image_ids.size()
				<< " for pyramid scale " << pyramid_scale << std::endl;
		}
	}
	std::cout << "Prepared " << files.size() << " scaled image files for pyramid scale "
		<< pyramid_scale << " in "
		<< std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
		<< " s" << std::endl;
	return files;
}

void RemoveScaleImageFiles(const std::vector<path> &files) {
	for (const auto &file : files) {
		if (exists(file)) remove(file);
	}
}

bool CheckImages(const std::vector<Problem> &problems) {
	if (problems.size() == 0) {
		return false;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return false;
	}
	const int width = image.cols;
	const int height = image.rows;
	for (size_t i = 1; i < problems.size(); ++i) {
		image_path = problems[i].dense_folder / path("images") / path(ToFormatIndex(problems[i].ref_image_id) + ".jpg");
		image = cv::imread(image_path.string());
		if (image.cols != width || image.rows != height) {
			return false;
		}
	}
	return true;
}

int ComputeRoundNum(const std::vector<Problem> &problems, const int initial_image_scale) {
	if (problems.size() == 0) {
		return 0;
	}
	path image_path = problems[0].dense_folder / path("images") / path(ToFormatIndex(problems[0].ref_image_id) + ".jpg");
	cv::Mat image = cv::imread(image_path.string());
	if (image.empty()) {
		return 0;
	}
	int max_size = MAX(image.cols, image.rows);
	max_size = MAX(1, (max_size + initial_image_scale / 2) / initial_image_scale);
	int round_num = 1;
	while (max_size > 1000) {
		max_size /= 2;
		round_num++;
	}
	return round_num;
}


void ProcessProblem(const Problem &problem) {
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	APD APD(problem);
	APD.InuputInitialization();
	APD.CudaSpaceInitialization();
	APD.SetDataPassHelperInCuda();
	APD.RunPatchMatch();

	int width = APD.GetWidth(), height = APD.GetHeight();
	cv::Mat depth = cv::Mat(height, width, CV_32FC1);
	cv::Mat normal = cv::Mat(height, width, CV_32FC3);
	cv::Mat pixel_states = APD.GetPixelStates();
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			float4 plane_hypothesis = APD.GetPlaneHypothesis(r, c);
			depth.at<float>(r, c) = plane_hypothesis.w;
			if (depth.at<float>(r, c) < APD.GetDepthMin() || depth.at<float>(r, c) > APD.GetDepthMax()) {
				depth.at<float>(r, c) = 0;
				pixel_states.at<uchar>(r, c) = UNKNOWN;
			}
			normal.at<cv::Vec3f>(r, c) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
		}
	}
	
	path depth_path = problem.result_folder / path("depths.dmb");
	WriteBinMat(depth_path, depth);
	path normal_path = problem.result_folder / path("normals.dmb");
	WriteBinMat(normal_path, normal);
	path weak_path = problem.result_folder / path("weak.bin");
	WriteBinMat(weak_path, pixel_states);
	path selected_view_path = problem.result_folder / path("selected_views.bin");
	WriteBinMat(selected_view_path, APD.GetSelectedViews());


	if (problem.show_medium_result) {
		path depth_img_path = problem.result_folder / path("depth_" + std::to_string(problem.iteration) + ".jpg");
		path normal_img_path = problem.result_folder / path("normal_" + std::to_string(problem.iteration) + ".jpg");
		path weak_img_path = problem.result_folder / path("weak_" + std::to_string(problem.iteration) + ".jpg");
		ShowDepthMap(depth_img_path, depth, APD.GetDepthMin(), APD.GetDepthMax());
		ShowNormalMap(normal_img_path, normal);
		ShowWeakImage(weak_img_path, pixel_states);
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Processing image: " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
	std::cout << "Cost time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}

int main(int argc, char **argv) {
	if (argc < 2) {
		std::cerr << "USAGE: APD dense_folder [gpu_index] [--initial-image-scale N] [--show-medium-result]\n";
		return EXIT_FAILURE;
	}
	path dense_folder(argv[1]);
	path output_folder = dense_folder / path("APD");
	create_directory(output_folder);
	// Keep the legacy positional GPU argument while allowing named options.
	int gpu_index = 0;
	int initial_image_scale = 4;
	bool show_medium_result = false;
	int arg = 2;
	if (arg < argc && std::string(argv[arg]).find("--") != 0) {
		gpu_index = std::atoi(argv[arg++]);
	}
	try {
		while (arg < argc) {
			const std::string option(argv[arg++]);
			if (option == "--initial-image-scale" && arg < argc) {
				const std::string value(argv[arg++]);
				size_t parsed = 0;
				initial_image_scale = std::stoi(value, &parsed);
				if (parsed != value.size() || initial_image_scale < 1) {
					throw std::invalid_argument("--initial-image-scale must be an integer >= 1 (1 = original resolution)");
				}
			} else if (option == "--show-medium-result") {
				show_medium_result = true;
			} else {
				throw std::invalid_argument("Unknown or incomplete option: " + option);
			}
		}
	} catch (const std::exception &error) {
		std::cerr << error.what() << "\n";
		return EXIT_FAILURE;
	}
	std::cout << "Initial image scale: 1/" << initial_image_scale
			  << " (width and height are each scaled by 1/" << initial_image_scale << ")\n";
	cudaSetDevice(gpu_index);
	// generate problems
	std::vector<Problem> problems;
	GenerateSampleList(dense_folder, problems);
	if (!CheckImages(problems)) {
		std::cerr << "Images may error, check it!\n";
		return EXIT_FAILURE;
	}
	for (auto &problem : problems) {
		problem.initial_image_scale = initial_image_scale;
		problem.show_medium_result = show_medium_result;
	}
	int num_images = problems.size();
	std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

	int round_num = ComputeRoundNum(problems, initial_image_scale);

	std::cout << "Round nums: " << round_num << std::endl;
	size_t image_cache_capacity = 1;
	for (const auto &problem : problems) {
		std::unordered_set<int> active(problem.src_image_ids.begin(), problem.src_image_ids.end());
		active.insert(problem.ref_image_id);
		image_cache_capacity = std::max(image_cache_capacity, active.size());
	}
	if (image_cache_capacity > MAX_IMAGES) {
		std::cerr << "A ProcessProblem requires " << image_cache_capacity
			<< " images, exceeding MAX_IMAGES=" << MAX_IMAGES << ".\n";
		return EXIT_FAILURE;
	}
	auto scaled_image_cache = std::make_shared<ScaledImageCache>(image_cache_capacity);
	for (auto &problem : problems) problem.scaled_image_cache = scaled_image_cache;
	std::cout << "Prepared-image cache capacity: " << image_cache_capacity
		<< " views (active views retained; each scale's temporary files are deleted "
		<< "when that scale finishes).\n";
	int iteration_index = 0;
	for (int i = 0; i < round_num; ++i) {
		const int pyramid_scale = static_cast<int>(std::pow(2, round_num - 1 - i));
		for (auto &problem : problems) {
			problem.scale_size = pyramid_scale;
		}
		const std::vector<path> scaled_image_files = PrepareScaleImages(
			problems, pyramid_scale, *scaled_image_cache);
		scaled_image_cache->ResetStats();
		for (auto &problem : problems) {
			{
				auto &params = problem.params;
				if (i == 0) {
					params.state = FIRST_INIT;
					params.use_APD = false;
				}
				else {
					params.state = REFINE_INIT;
					params.use_APD = true;
					params.ransac_threshold = 0.01 - i * 0.00125;
					params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
				}
				params.geom_consistency = false;
				params.max_iterations = 3;
				params.weak_peak_radius = 6;
			}
			problem.iteration = iteration_index;
			problem.show_medium_result = show_medium_result;
			problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale
			ProcessProblem(problem);
		}
		iteration_index++;
		for (int j = 0; j < 3; ++j) {
			for (auto &problem : problems) {
				{
					auto &params = problem.params;
					params.state = REFINE_ITER;
					if (i == 0) {
						params.use_APD = false;
					}
					else {
						params.use_APD = true;
						params.ransac_threshold = 0.01 - i * 0.00125;
						params.rotate_time = MIN(static_cast<int>(std::pow(2, i)), 4);
					}
					params.geom_consistency = true;
					params.max_iterations = 3;
					params.weak_peak_radius = MAX(4 - 2 * j, 2);
				}
				problem.iteration = iteration_index;
				problem.show_medium_result = show_medium_result;
				problem.scale_size = static_cast<int>(std::pow(2, round_num - 1 - i)); // scale
				ProcessProblem(problem);
			}
			iteration_index++;
		}
		std::cout << "Round: " << i << " done\n";
		std::cout << "Scaled-image cache at pyramid scale " << pyramid_scale << ": "
			<< scaled_image_cache->CacheHits() << " hits, "
			<< scaled_image_cache->CacheMisses() << " temp-file reads, "
			<< scaled_image_cache->DiskReadSeconds() << " s reading scaled images.\n";
		scaled_image_cache->Clear();
		RemoveScaleImageFiles(scaled_image_files);
		std::cout << "Pyramid scale " << pyramid_scale << " temporary scaled images deleted.\n";
	}

	RunFusion(dense_folder, problems);
	{// delete files
		for (size_t i = 0; i < problems.size(); ++i) {
			const auto &problem = problems[i];
			remove(problem.result_folder / path("weak.bin"));
			// Keep final depth and normal maps for downstream reconstruction tools.
			remove(problem.result_folder / path("selected_views.bin"));
			//remove(problem.result_folder / path("neighbour.bin")); 
			//remove(problem.result_folder / path("neighbour_map.bin"));
		}
	}
	std::cout << "All done\n";
	return EXIT_SUCCESS;
}