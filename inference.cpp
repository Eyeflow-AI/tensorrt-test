// #include <iostream>
#include <string>
#include <vector>
// #include <cmath>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "trt_utils.h"
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

    LOG(INFO) << "Engine file path: " << engine_file_path << std::endl;

    int patch_size = 160;
    // int max_image_side = 1600;
    int max_input_side = 2000;

    CustomLogger logger{};

    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> m_runtime{nvinfer1::createInferRuntime(logger)};
    if (m_runtime == nullptr)
    {
        LOG(ERROR) << "Failed to create the runtime." << std::endl;
        return EXIT_FAILURE;
    }

    // Load the plugin library.
    initLibNvInferPlugins(&logger, "");
    m_runtime->getPluginRegistry().loadLibrary("./build/libtensorrt_plugin-tile_image.so");
    m_runtime->getPluginRegistry().loadLibrary("./build/libtensorrt_plugin-adjust_tiled_boxes.so");

    LOG(INFO) << "Loading engine file: " << engine_file_path << std::endl;
    std::ifstream engine_file{engine_file_path, std::ios::binary};
    if (!engine_file)
    {
        LOG(ERROR) << "Failed to open the engine file." << std::endl;
        return EXIT_FAILURE;
    }

    engine_file.seekg(0, std::ios::end);
    size_t const engine_file_size{static_cast<size_t>(engine_file.tellg())};
    engine_file.seekg(0, std::ios::beg);

    std::unique_ptr<char[]> engine_data{new char[engine_file_size]};
    engine_file.read(engine_data.get(), engine_file_size);

	std::shared_ptr<nvinfer1::ICudaEngine> m_engine; //!< The TensorRT engine used to run the network
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_data.get(), engine_file_size), InferDeleter());
    if (!m_engine)
    {
        std::string errmsg = "Error creating TRT Engine";
        LOG(ERROR) << errmsg;
        throw std::runtime_error(errmsg);
    }

    // Create the execution context.
    std::shared_ptr<nvinfer1::IExecutionContext> m_context{m_engine->createExecutionContext()};
    if (m_context == nullptr)
    {
        LOG(ERROR) << "Failed to create the execution context." << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat orig_image = cv::imread(image_file_path);
    cv::Mat input_image;
    double resize_scale = std::min(static_cast<double>(max_input_side) / orig_image.cols, static_cast<double>(max_input_side) / orig_image.rows);
    if (resize_scale < 1.0)
        cv::resize(orig_image, input_image, cv::Size(), resize_scale, resize_scale, cv::INTER_LINEAR);
    // orig_image(cv::Range(500, 500 + 2 * patch_size), cv::Range(3100, 3100 + 2 * patch_size)).convertTo(input_image, CV_8UC3);
    // orig_image(cv::Range(500, 500 + 2 * patch_size), cv::Range(1000, 1000 + 2 * patch_size)).convertTo(input_image, CV_8UC3);
    int m_input_channels = 3;

    auto cv_input_depth_size = CV_8UC3;
    if (m_input_channels == 1)
    {
        cv::cvtColor(input_image, input_image, cv::COLOR_RGB2GRAY);
        cv_input_depth_size = CV_8UC1;
    }


    int num_classes = 1;
    std::map<std::string, std::vector<uint>> max_dims{
        {"input", {1, (uint)max_input_side, (uint)max_input_side, 1}},
        {"input_image_size", {4}},
        // {"output_tiles", {56, (uint)patch_size, (uint)patch_size, 1}},
        // {"output_boxes", {1, 1904000, 4}},
        // {"output_classification", {1, 1904000, (uint)num_classes}},
        // {"decode_boxes_1", {56, 34000, (uint)num_classes}},
    };
    auto m_buffers = std::make_unique<BufferManager>(m_engine, max_dims, m_context);

    uint32_t* img_size_buffer = static_cast<uint32_t*>(m_buffers->get_host_buffer("input_image_size"));
    img_size_buffer[0] = 1; // batch size
    img_size_buffer[1] = input_image.rows; // height
    img_size_buffer[2] = input_image.cols; // width
    img_size_buffer[3] = m_input_channels; // channels
    size_t input_img_size_dtype_byte_size = sizeof(uint32_t);
    size_t const input_img_size_tensor_size_bytes{4 * input_img_size_dtype_byte_size};
    std::string input_img_size_tensor_name{"input_image_size"};
    m_buffers->copy_input_to_device_async(input_img_size_tensor_name, input_img_size_tensor_size_bytes, stream);
    m_buffers->set_input_shape(input_img_size_tensor_name, std::vector<int>({4}));

    int m_output_channels = 1;
    auto cv_output_depth_size = CV_32FC3;
    if (m_output_channels == 1)
    {
        cv_output_depth_size = CV_32FC1;
    }

    uint8_t* input_image_buffer = static_cast<uint8_t*>(m_buffers->get_host_buffer("input"));
    cv::Mat dest_image(cv::Size(input_image.cols, input_image.rows), cv_input_depth_size, (void*)(static_cast<uint8_t*>(input_image_buffer)));
    input_image.convertTo(dest_image, cv_input_depth_size);
    cv::imwrite("./build/input_image.jpg", dest_image);

    size_t input_image_dtype_byte_size = sizeof(uint8_t);
    size_t const input_tensor_size_bytes{input_image.cols* input_image.rows * m_input_channels * input_image_dtype_byte_size};
    std::string input_tensor_name{"input"};
    m_buffers->copy_input_to_device_async(input_tensor_name, input_tensor_size_bytes, stream);
    m_buffers->set_input_shape(input_tensor_name, std::vector<int>({1, input_image.rows, input_image.cols, m_input_channels}));

    // Run inference a couple of times.
    bool const status{m_context->enqueueV3(stream)};
    if (!status)
    {
        LOG(ERROR) << "Failed to run inference." << std::endl;
        return EXIT_FAILURE;
    }

    m_buffers->copy_output_to_host_async(stream);


    int m_max_boxes = 30;
    double m_confidence_thresh = 0.05;
    auto num_detections = m_buffers->get_mat("num_detections");
    auto predicted_boxes = m_buffers->get_mat("detection_boxes");
    auto predicted_scores = m_buffers->get_mat("detection_scores");
    auto predicted_labels = m_buffers->get_mat("detection_classes");

    // auto output_tiles = m_buffers->get_mat("output_tiles");
    // auto input_image_size = m_buffers->get_mat("input_image_size");
    // auto output_boxes = m_buffers->get_mat("output_boxes");
    // auto output_classification = m_buffers->get_mat("decode_boxes_1");
    // auto output_classification = m_buffers->get_mat("output_classification");

    // float* output_patch_buffer = static_cast<float*>(m_buffers->get_host_buffer("output_tiles"));
    // size_t output_patch_dtype_byte_size = sizeof(float);
    // size_t const output_patch_size_bytes{m_output_channels * patch_size * patch_size * output_patch_dtype_byte_size};
    // for (int patch = 0; patch < 6; ++patch)
    // {
    //     cv::Mat output_patch(cv::Size(patch_size, patch_size), cv_output_depth_size, (char*)output_patch_buffer + (patch * output_patch_size_bytes));

    //     auto save_ret = cv::imwrite("./build/tile_image_" + std::to_string(patch) + ".jpg", output_patch);
    //     if (!save_ret)
    //         throw std::runtime_error("Fail to save patch");
    // }


    json annotations;
    std::list<json> instances;
    std::vector<json> inst_annot;
    for (int det = 0; det < num_detections.at<int32_t>(0, 0); det++)
    {
        if (det > m_max_boxes)
            break;

            float score = predicted_scores.at<float>(0, det);
        if (score < m_confidence_thresh)
            continue;

        int det_class = (int)predicted_labels.at<int32_t>(0, det);

        double x_min = (double)predicted_boxes.at<float>(0, det, 0) / resize_scale;
        double y_min = (double)predicted_boxes.at<float>(0, det, 1) / resize_scale;
        double x_max = (double)predicted_boxes.at<float>(0, det, 2) / resize_scale;
        double y_max = (double)predicted_boxes.at<float>(0, det, 3) / resize_scale;
        // double x_min = (double)predicted_boxes.at<float>(0, det, 0);
        // double y_min = (double)predicted_boxes.at<float>(0, det, 1);
        // double x_max = (double)predicted_boxes.at<float>(0, det, 2);
        // double y_max = (double)predicted_boxes.at<float>(0, det, 3);

        int width = 1;
        cv::Scalar color(0, 255, 0);
        // cv::rectangle(input_image, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min), color, width);
        cv::rectangle(orig_image, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min), color, width);

        instances.push_back({
            {"class", "furo"},
            {"label", "furo"},
            {"bbox", {
                {"x_min", (int)x_min},
                {"y_min", (int)y_min},
                {"x_max", (int)x_max},
                {"y_max", (int)y_max}
            }},
            {"color", "#ffffff"},
            {"confidence", round(score * 1000) / 1000}
        });
    }

    // cv::imwrite("./build/output_image.jpg", input_image);
    cv::imwrite("./build/output_image.jpg", orig_image);

    // if (instances.size() > 0)
    //     instances = join_detections(instances);


    // Release resources.
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return 0;
}
