#include <string>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <glog/logging.h>

#include <opencv2/opencv.hpp>

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "trt_utils_old.h"
#include "patch_pos.h"
// --------------------------------------------------------------------------------------------------------------------------------


inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? ", " : "") << dims.d[i];
    }
    return os << ")";
}
// --------------------------------------------------------------------------------------------------------------------------------


class TestModelOld
{
private:
	trt_utils::Logger* m_logger;
	trt_utils::InferUniquePtr<nvinfer1::IRuntime> m_runtime;
	std::shared_ptr<nvinfer1::ICudaEngine> m_engine; //!< The TensorRT engine used to run the network
	trt_utils::InferUniquePtr<nvinfer1::IExecutionContext> m_context;
	std::unique_ptr<trt_utils::BufferManager> m_buffers;
    std::shared_ptr<nvinfer1::ICudaEngine> m_nms_engine;
    trt_utils::InferUniquePtr<nvinfer1::IExecutionContext> m_nms_context;
    std::unique_ptr<trt_utils::BufferManager> m_nms_buffers;

	nvinfer1::Dims m_input_dims;  //!< The dimensions of the input to the network.
	std::vector<nvinfer1::Dims> m_output_dims; //!< The dimensions of the output to the network.
	std::vector<std::string> m_input_tensor_names;
	std::vector<std::string> m_output_tensor_names;

    std::vector<nvinfer1::Dims> m_nms_input_dims;  //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> m_nms_output_dims; //!< The dimensions of the output to the network.
    std::vector<std::string> m_nms_input_tensor_names;
    std::vector<std::string> m_nms_output_tensor_names;


    int m_input_channels;
    int m_max_boxes;
    int m_patch_size;
    uint m_max_batch_size;
    int m_max_num_patches_frame;
    double m_patch_pos_resize_scale;

public:
    TestModelOld()
    {
        LOG(INFO) << "TestModelOld initialized." << std::endl;
        m_input_channels = 3;
        m_max_boxes = 30;
        m_patch_size = 160;
        m_max_batch_size = 32;
        m_max_num_patches_frame = 48;
    };


    ~TestModelOld()
    {
    };

    void adjust_boxes(cv::Mat &boxes, PatchPos& patch_pos)
    {
        cv::Mat adjust_grid = patch_pos.get_image_grid();
        adjust_grid = cv::repeat(adjust_grid, 1, boxes.size[1]);
        int reshape[] = {boxes.size[0], boxes.size[1], 4};
        cv::Mat add_grid(3, reshape, adjust_grid.type(), adjust_grid.data);
        cv::add(boxes, add_grid, boxes);
    }

    void load_model(const std::string& engine_file_path)
    {
        // LOG(INFO) << "Engine file path: " << engine_file_path << std::endl;
        fs::path trt_model_file("./build/roi_location_old.trt");
        if (!fs::is_regular_file(trt_model_file))
            throw std::runtime_error("TRT Model not found: " + (std::string)trt_model_file + " - Need to convert the ONNX model to TRT first.");

        fs::path trt_nms_model_file = "./build/roi_location_old_nms.trt";
        if (!fs::is_regular_file(trt_nms_model_file))
            throw std::runtime_error("TRT Model not found: " + (std::string)trt_nms_model_file + " - Need to convert the ONNX model to TRT first.");

        m_logger = trt_utils::Logger::get_instance();
        initLibNvInferPlugins((void*)m_logger, "");

        m_runtime = trt_utils::InferUniquePtr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger->getTRTLogger())};
        if (!m_runtime)
        {
            std::string errmsg = "Error creating TRT Runtime";
            LOG(ERROR) << errmsg;
            throw std::runtime_error(errmsg);
        }

        std::vector<char> plan;
        std::ifstream ifs((std::string)trt_model_file, std::ios::in | std::ios::binary);
        if (!ifs.is_open())
            throw std::runtime_error("Fail open TRT Model: " + (std::string)trt_model_file);

        ifs.seekg(0, ifs.end);
        size_t size = ifs.tellg();
        plan.resize(size);
        ifs.seekg(0, ifs.beg);
        ifs.read(plan.data(), size);
        ifs.close();

        m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan.data(), plan.size()), trt_utils::InferDeleter());
        if (!m_engine)
        {
            std::string errmsg = "Error creating TRT Engine";
            LOG(ERROR) << errmsg;
            throw std::runtime_error(errmsg);
        }

        m_context = trt_utils::InferUniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (!m_context)
        {
            std::string errmsg = "Error creating TRT Context";
            LOG(ERROR) << errmsg;
            throw std::runtime_error(errmsg);
        }

        // Create RAII buffer manager object
        m_buffers = std::make_unique<trt_utils::BufferManager>(m_engine, std::vector<uint>({m_max_batch_size}));

        int nb_bindings = static_cast<int>(m_engine->getNbIOTensors());
        for (int i = 0; i < nb_bindings; i++)
        {
            const char *name = m_engine->getIOTensorName(i);
            auto tensor_io = m_engine->getTensorIOMode(name);
            if (tensor_io == nvinfer1::TensorIOMode::kINPUT)
            {
                m_input_dims = m_engine->getTensorShape(name);
                m_input_tensor_names.push_back(name);
                LOG(INFO) << "Input: " << name << " - " << m_input_dims;
            }
            else
            {
                m_output_dims.push_back(m_engine->getTensorShape(name));
                m_output_tensor_names.push_back(name);
                LOG(INFO) << "Output: " << name << " - " << m_output_dims.back();
            }
        }

        // load the nms additional engine
        std::vector<char> plan_nms;
        std::ifstream ifs_nms(trt_nms_model_file, std::ios::in | std::ios::binary);
        if (!ifs_nms.is_open())
            throw std::runtime_error("Fail open TRT NMS Model: " + (std::string)trt_nms_model_file);

        ifs_nms.seekg(0, ifs_nms.end);
        size = ifs_nms.tellg();
        plan_nms.resize(size);
        ifs_nms.seekg(0, ifs_nms.beg);
        ifs_nms.read(plan_nms.data(), size);
        ifs_nms.close();

        m_nms_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan_nms.data(), plan_nms.size()), trt_utils::InferDeleter());
        if (!m_nms_engine)
        {
            std::string errmsg = "Error creating TRT Engine";
            LOG(ERROR) << errmsg;
            throw std::runtime_error(errmsg);
        }

        m_nms_context = trt_utils::InferUniquePtr<nvinfer1::IExecutionContext>(m_nms_engine->createExecutionContext());
        if (!m_nms_context)
        {
            std::string errmsg = "Error creating TRT Context";
            LOG(ERROR) << errmsg;
            throw std::runtime_error(errmsg);
        }

        // Create RAII buffer manager object
        m_nms_buffers = std::make_unique<trt_utils::BufferManager>(m_nms_engine, std::vector<uint>({1, m_max_num_patches_frame * (uint)m_output_dims[0].d[1] * (uint)m_output_dims[0].d[2]}));

        nb_bindings = static_cast<int>(m_nms_engine->getNbIOTensors());
        for (int i = 0; i < nb_bindings; i++)
        {
            const char *name = m_nms_engine->getIOTensorName(i);
            auto tensor_io = m_nms_engine->getTensorIOMode(name);
            if (tensor_io == nvinfer1::TensorIOMode::kINPUT)
            {
                m_nms_input_dims.push_back(m_nms_engine->getTensorShape(name));
                m_nms_input_tensor_names.push_back(name);
                LOG(INFO) << "Input: " << name << " - " << m_nms_input_dims.back();
            }
            else
            {
                m_nms_output_dims.push_back(m_nms_engine->getTensorShape(name));
                m_nms_output_tensor_names.push_back(name);
                LOG(INFO) << "Output: " << name << " - " << m_nms_output_dims.back();
            }
        }

        m_input_channels = m_input_dims.d[3];
        m_max_boxes = std::min(m_max_boxes, (int)m_nms_output_dims[1].d[1]);

        // warm up
        int count = 10;
        while (count-- > 0)
        {
            auto dims = m_input_dims;
            dims.d[0] = 1;
            auto ret = m_context->setInputShape(m_input_tensor_names[0].c_str(), dims);
            if (!ret)
            {
                std::string errmsg = "Error running model: ";
                LOG(ERROR) << errmsg;
                throw std::runtime_error(errmsg);
            }

            bool status = m_context->executeV2(m_buffers->getDeviceBindings().data());
            if (!status)
            {
                std::string errmsg = "Error running model: ";
                LOG(ERROR) << errmsg;
                throw std::runtime_error(errmsg);
            }
        }


    };


    void run_inference(const cv::Mat &input_image, int max_image_side)
    {
        try
        {
            void* host_data_buffer = m_buffers->getHostBuffer(m_input_tensor_names[0]);
            auto element_size = m_buffers->getBufferElementSize(m_input_tensor_names[0]);
            PatchPos patch_pos(m_patch_size, max_image_side, m_input_channels, input_image, host_data_buffer, element_size, m_max_batch_size);
            m_patch_pos_resize_scale = patch_pos.m_resize_scale;

            uint batch_size = std::min(patch_pos.m_num_patches_total, m_max_batch_size);
            cv::Mat boxes, classification;
            for (uint batch_idx = 0; batch_idx < patch_pos.m_num_patches_total;)
            {
                patch_pos.fill_slices_batch(batch_idx, batch_idx + batch_size);

                auto dims = m_input_dims;
                dims.d[0] = batch_size;
                auto ret = m_context->setInputShape(m_input_tensor_names[0].c_str(), dims);
                if (!ret)
                {
                    std::string errmsg = "Error running model: ";
                    LOG(ERROR) << errmsg;
                    throw std::runtime_error(errmsg);
                }

                // Memcpy from host input buffers to device input buffers
                m_buffers->copyInputToDevice();

                bool status = m_context->executeV2(m_buffers->getDeviceBindings().data());
                if (!status)
                {
                    std::string errmsg = "Error running model: ";
                    LOG(ERROR) << errmsg;
                    throw std::runtime_error(errmsg);
                }

                // Memcpy from device output buffers to host output buffers
                m_buffers->copyOutputToHost();

                auto decode_boxes = m_buffers->getMat("decode_boxes", batch_size);
                auto decode_classification = m_buffers->getMat("decode_boxes_1", batch_size);
                if(boxes.size[0] == 0)
                {
                    decode_boxes.convertTo(boxes, decode_boxes.type());
                    decode_classification.convertTo(classification, decode_classification.type());
                }
                else
                {
                    // append boxes
                    cv::Mat dest_boxes(
                        std::vector<int>({boxes.size[0] + decode_boxes.size[0], boxes.size[1], boxes.size[2]}),
                        boxes.type()
                    );

                    int boxes_size = boxes.size[0] * boxes.size[1] * boxes.size[2] * sizeof(float);
                    int decode_boxes_size = decode_boxes.size[0] * decode_boxes.size[1] * decode_boxes.size[2] * sizeof(float);
                    memcpy((void*)dest_boxes.data, (void*)boxes.data, boxes_size);
                    memcpy((void*)((uint8_t*)dest_boxes.data + boxes_size), (void*)decode_boxes.data, decode_boxes_size);

                    boxes = std::move(dest_boxes);

                    // append classifications
                    cv::Mat dest_classification(
                        std::vector<int>({classification.size[0] + decode_classification.size[0], classification.size[1], classification.size[2]}),
                        classification.type()
                    );

                    int classification_size = classification.size[0] * classification.size[1] * classification.size[2] * sizeof(float);
                    int decode_classification_size = decode_classification.size[0] * decode_classification.size[1] * decode_classification.size[2] * sizeof(float);
                    memcpy((void*)dest_classification.data, (void*)classification.data, classification_size);
                    memcpy((void*)((uint8_t*)dest_classification.data + classification_size), (void*)decode_classification.data, decode_classification_size);

                    classification = std::move(dest_classification);
                }

                batch_idx += batch_size;
                batch_size = std::min(patch_pos.m_num_patches_total - batch_idx, m_max_batch_size);
            }

            adjust_boxes(boxes, patch_pos);

            void* host_box_buffer = m_nms_buffers->getHostBuffer(m_nms_input_tensor_names[0]);
            cv::Mat input_boxes(
                std::vector<int>({boxes.size[0], boxes.size[1], boxes.size[2]}),
                boxes.type(),
                host_box_buffer
            );

            boxes.convertTo(input_boxes, boxes.type());

            void* host_classification_buffer = m_nms_buffers->getHostBuffer(m_nms_input_tensor_names[1]);
            cv::Mat input_classification(
                std::vector<int>({classification.size[0], classification.size[1], classification.size[2]}),
                classification.type(),
                host_classification_buffer
            );

            classification.convertTo(input_classification, classification.type());

            auto dims = m_nms_input_dims[0];
            dims.d[0] = 1;
            dims.d[1] = boxes.size[0] * boxes.size[1];
            auto ret = m_nms_context->setInputShape(m_nms_input_tensor_names[0].c_str(), dims);

            dims = m_nms_input_dims[1];
            dims.d[0] = 1;
            dims.d[1] = classification.size[0] * classification.size[1];
            ret = m_nms_context->setInputShape(m_nms_input_tensor_names[1].c_str(), dims);

            // Memcpy from host input buffers to device input buffers
            m_nms_buffers->copyInputToDevice();

            bool status = m_nms_context->executeV2(m_nms_buffers->getDeviceBindings().data());
            if (!status)
            {
                std::string errmsg = "Error running model: ";
                LOG(ERROR) << errmsg;
                throw std::runtime_error(errmsg);
            }

            // Memcpy from device output buffers to host output buffers
            m_nms_buffers->copyOutputToHost();

            return;
        }
        catch(std::exception& excpt)
        {
            LOG(ERROR) << "Fail processing frame" << excpt.what();
            return;
        }
        catch(...)
        {
            LOG(ERROR) << "Fail processing flow";
            return;
        }
    };


    void draw_output(const cv::Mat &input_image, double resize_scale, std::string output_path)
    {
        double m_confidence_thresh = 0.05;
        auto num_detections = m_nms_buffers->getMat("num_detections");
        auto predicted_boxes = m_nms_buffers->getMat("detection_boxes");
        auto predicted_scores = m_nms_buffers->getMat("detection_scores");
        auto predicted_labels = m_nms_buffers->getMat("detection_classes");

        for (int det = 0; det < num_detections.at<int32_t>(0, 0); det++)
        {
            if (det > m_max_boxes)
                break;

                float score = predicted_scores.at<float>(0, det);
            if (score < m_confidence_thresh)
                continue;

            int det_class = (int)predicted_labels.at<int32_t>(0, det);

            double x_min = (double)predicted_boxes.at<float>(0, det, 0) / m_patch_pos_resize_scale;
            double y_min = (double)predicted_boxes.at<float>(0, det, 1) / m_patch_pos_resize_scale;
            double x_max = (double)predicted_boxes.at<float>(0, det, 2) / m_patch_pos_resize_scale;
            double y_max = (double)predicted_boxes.at<float>(0, det, 3) / m_patch_pos_resize_scale;

            x_min /= resize_scale;
            y_min /= resize_scale;
            x_max /= resize_scale;
            y_max /= resize_scale;

            int width = 1;
            cv::Scalar color(0, 255, 0);
            cv::rectangle(input_image, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min), color, width);
        }

        cv::imwrite(output_path, input_image);
    };

};