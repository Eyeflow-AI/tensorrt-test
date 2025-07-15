#include <string>
#include <fstream>
// #include <filesystem>
// namespace fs = std::filesystem;

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "trt_utils.h"
// --------------------------------------------------------------------------------------------------------------------------------

class TestModelNew
{
private:
    CustomLogger m_logger;
    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> m_runtime;
    cudaStream_t m_stream;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine; //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    std::unique_ptr<BufferManager> m_buffers;
    int m_input_channels;
    int m_max_boxes;
    int m_patch_size;

public:
    TestModelNew():
        m_runtime{nvinfer1::createInferRuntime(m_logger)}
    {
        LOG(INFO) << "TestModelNew initialized." << std::endl;
        m_input_channels = 3;
        m_max_boxes = 30;
        m_patch_size = 160;
    };


    ~TestModelNew()
    {
        // Release resources.
        CHECK_CUDA_ERROR(cudaStreamDestroy(m_stream));
    };


    void load_model(const std::string& engine_file_path, int max_input_side)
    {
        LOG(INFO) << "Engine file path: " << engine_file_path << std::endl;

        // CustomLogger m_logger{};

        // Create CUDA stream.
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));

        if (m_runtime == nullptr)
        {
            LOG(ERROR) << "Failed to create the runtime." << std::endl;
            return;
        }

        // Load the plugin library.
        initLibNvInferPlugins(&m_logger, "");
        m_runtime->getPluginRegistry().loadLibrary("./build/libtensorrt_plugin-tile_image.so");
        m_runtime->getPluginRegistry().loadLibrary("./build/libtensorrt_plugin-adjust_tiled_boxes.so");

        LOG(INFO) << "Loading engine file: " << engine_file_path << std::endl;
        std::ifstream engine_file{engine_file_path, std::ios::binary};
        if (!engine_file)
        {
            LOG(ERROR) << "Failed to open the engine file." << std::endl;
            return;
        }

        engine_file.seekg(0, std::ios::end);
        size_t const engine_file_size{static_cast<size_t>(engine_file.tellg())};
        engine_file.seekg(0, std::ios::beg);

        std::unique_ptr<char[]> engine_data{new char[engine_file_size]};
        engine_file.read(engine_data.get(), engine_file_size);

        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_data.get(), engine_file_size));
        if (!m_engine)
        {
            std::string errmsg = "Error creating TRT Engine";
            LOG(ERROR) << errmsg;
            throw std::runtime_error(errmsg);
        }

        // Create the execution context.
        m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (m_context == nullptr)
        {
            LOG(ERROR) << "Failed to create the execution context." << std::endl;
            return;
        }

        int num_classes = 1;
        std::map<std::string, std::vector<uint>> max_dims{
            {"input", {1, (uint)max_input_side, (uint)max_input_side, 3}},
            {"input_image_size", {4}},
        };

        m_buffers = std::make_unique<BufferManager>(m_engine, max_dims, m_context);


        // warm up
        int count = 10;
        while (count-- > 0)
        {
            m_buffers->set_input_shape("input", std::vector<int>({1, 2000, 2000, 3}));

            bool status = m_context->enqueueV3(m_stream);
            if (!status)
            {
                std::string errmsg = "Error running model";
                LOG(ERROR) << errmsg;
                throw std::runtime_error(errmsg);
            }
        }

    };


    void run_inference(const cv::Mat &input_image)
    {
        uint32_t* img_size_buffer = static_cast<uint32_t*>(m_buffers->get_host_buffer("input_image_size"));
        img_size_buffer[0] = 1; // batch size
        img_size_buffer[1] = input_image.rows; // height
        img_size_buffer[2] = input_image.cols; // width
        img_size_buffer[3] = m_input_channels; // channels
        size_t input_img_size_dtype_byte_size = sizeof(uint32_t);
        size_t const input_img_size_tensor_size_bytes{4 * input_img_size_dtype_byte_size};
        std::string input_img_size_tensor_name{"input_image_size"};
        m_buffers->copy_input_to_device_async(input_img_size_tensor_name, input_img_size_tensor_size_bytes, m_stream);
        m_buffers->set_input_shape(input_img_size_tensor_name, std::vector<int>({4}));

        auto cv_input_depth_size = CV_8UC3;
        if (m_input_channels == 1)
        {
            cv::cvtColor(input_image, input_image, cv::COLOR_RGB2GRAY);
            cv_input_depth_size = CV_8UC1;
        }

        uint8_t* input_image_buffer = static_cast<uint8_t*>(m_buffers->get_host_buffer("input"));
        cv::Mat dest_image(cv::Size(input_image.cols, input_image.rows), cv_input_depth_size, (void*)(static_cast<uint8_t*>(input_image_buffer)));
        input_image.convertTo(dest_image, cv_input_depth_size);
        // cv::imwrite("./build/input_image.jpg", dest_image);

        size_t input_image_dtype_byte_size = sizeof(uint8_t);
        size_t const input_tensor_size_bytes{input_image.cols* input_image.rows * m_input_channels * input_image_dtype_byte_size};
        std::string input_tensor_name{"input"};
        m_buffers->copy_input_to_device_async(input_tensor_name, input_tensor_size_bytes, m_stream);
        m_buffers->set_input_shape(input_tensor_name, std::vector<int>({1, input_image.rows, input_image.cols, m_input_channels}));

        // Run inference a couple of times.
        bool const status{m_context->enqueueV3(m_stream)};
        if (!status)
        {
            LOG(ERROR) << "Failed to run inference." << std::endl;
            return;
        }

        m_buffers->copy_output_to_host_async(m_stream);
    };


    void draw_output(const cv::Mat &input_image, double resize_scale, std::string output_path)
    {
        double m_confidence_thresh = 0.05;
        auto num_detections = m_buffers->get_mat("num_detections");
        auto predicted_boxes = m_buffers->get_mat("detection_boxes");
        auto predicted_scores = m_buffers->get_mat("detection_scores");
        auto predicted_labels = m_buffers->get_mat("detection_classes");

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

            int width = 1;
            cv::Scalar color(0, 255, 0);
            cv::rectangle(input_image, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min), color, width);
        }

        cv::imwrite(output_path, input_image);
    };

};