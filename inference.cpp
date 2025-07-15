// #include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "new_model.cpp"
#include "old_model.cpp"
// --------------------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        LOG(ERROR) << "Usage: " << argv[0]
                  << "<engine_file_path> <image_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string const engine_file_path{argv[1]};
    std::string const image_file_path{argv[2]};

    // int patch_size = 160;
    int max_image_side = 1600;
    int max_input_side = 1600;

    cv::Mat orig_image = cv::imread(image_file_path);
    cv::Mat input_image;
    double resize_scale = std::min(1.0, std::min(static_cast<double>(max_input_side) / orig_image.cols, static_cast<double>(max_input_side) / orig_image.rows));
    if (resize_scale < 1.0)
        cv::resize(orig_image, input_image, cv::Size(), resize_scale, resize_scale, cv::INTER_LINEAR);
    // orig_image(cv::Range(500, 500 + 2 * patch_size), cv::Range(3100, 3100 + 2 * patch_size)).convertTo(input_image, CV_8UC3);
    // orig_image(cv::Range(500, 500 + 2 * patch_size), cv::Range(1000, 1000 + 2 * patch_size)).convertTo(input_image, CV_8UC3);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    long long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();



    TestModelOld old_model;
    old_model.load_model(engine_file_path);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
    {
        old_model.run_inference(input_image, max_image_side);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    LOG(INFO) << "Inference time for old model: " << milliseconds / 100.0 << " ms" << std::endl;

    old_model.draw_output(orig_image, resize_scale, "./build/output_image_old.jpg");


    TestModelNew new_model;
    new_model.load_model(engine_file_path, max_input_side);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i)
    {
        new_model.run_inference(input_image);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    LOG(INFO) << "Inference time for new model: " << milliseconds / 100.0 << " ms" << std::endl;

    new_model.draw_output(orig_image, resize_scale, "./build/output_image_new.jpg");

    return 0;
}
