#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include <opencv2/opencv.hpp>

#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
// --------------------------------------------------------------------------------------------------------------------------------

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
// --------------------------------------------------------------------------------------------------------------------------------

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
// --------------------------------------------------------------------------------------------------------------------------------

class CustomLogger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};
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


int main(int argc, char* argv[])
{
    std::cout << "Load model";

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << "<engine_file_path> <image_file_path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string const engine_file_path{argv[1]};
    std::string const image_file_path{argv[2]};

    std::cout << "Engine file path: " << engine_file_path << std::endl;

    int m_input_channels = 3;
    int patch_size = 320;
    int max_input_size = 5000;
    int max_side_size = 1920;

	std::vector<nvinfer1::Dims> m_input_dims;  //!< The dimensions of the input to the network.
	std::vector<nvinfer1::Dims> m_output_dims; //!< The dimensions of the output to the network.
	std::vector<std::string> m_input_tensor_names;
	std::vector<std::string> m_output_tensor_names;

    CustomLogger logger{};

    // Create CUDA stream.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> m_runtime{nvinfer1::createInferRuntime(logger)};
    if (m_runtime == nullptr)
    {
        std::cerr << "Failed to create the runtime." << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream engine_file{engine_file_path, std::ios::binary};
    if (!engine_file)
    {
        std::cerr << "Failed to open the engine file." << std::endl;
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
        std::cerr << errmsg;
        throw std::runtime_error(errmsg);
    }

    // Create the execution context.
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> m_context{m_engine->createExecutionContext()};
    if (m_context == nullptr)
    {
        std::cerr << "Failed to create the execution context." << std::endl;
        return EXIT_FAILURE;
    }

    nvinfer1::TensorFormat const expected_format{nvinfer1::TensorFormat::kLINEAR};

    // IO tensor information and buffers.
    std::vector<nvinfer1::Dims> input_tensor_shapes{};
    std::vector<nvinfer1::Dims> output_tensor_shapes{};
    std::vector<size_t> input_tensor_sizes{};
    std::vector<size_t> output_tensor_sizes{};
    std::vector<char const*> input_tensor_names{};
    std::vector<char const*> output_tensor_names{};
    std::vector<void*> input_tensor_host_buffers{};
    std::vector<void*> input_tensor_device_buffers{};
    std::vector<void*> output_tensor_host_buffers{};
    std::vector<void*> output_tensor_device_buffers{};

    // Check the number of IO tensors.
    int32_t const num_io_tensors{m_engine->getNbIOTensors()};
    std::cout << "Number of IO Tensors: " << num_io_tensors << std::endl;
    for (int32_t i{0}; i < num_io_tensors; ++i)
    {
        char const* const tensor_name{m_engine->getIOTensorName(i)};
        std::cout << "Tensor name: " << tensor_name << std::endl;
        nvinfer1::TensorIOMode const io_mode{m_engine->getTensorIOMode(tensor_name)};
        nvinfer1::DataType const dtype{m_engine->getTensorDataType(tensor_name)};
        size_t expected_dtype_byte_size;
        if (dtype == nvinfer1::DataType::kFLOAT)
        {
            expected_dtype_byte_size = 4;
        }
        else if (dtype == nvinfer1::DataType::kINT32)
        {
            expected_dtype_byte_size = 4;
        }
        else if (dtype == nvinfer1::DataType::kINT64)
        {
            expected_dtype_byte_size = 8;
        }
        else
        {
            std::cerr << "Invalid data type." << std::endl;
            return EXIT_FAILURE;
        }

        nvinfer1::TensorFormat const format{m_engine->getTensorFormat(tensor_name)};
        if (format != expected_format)
        {
            std::cerr << "Invalid tensor format." << std::endl;
            return EXIT_FAILURE;
        }
        // Because the input and output shapes are static,
        // there is no need to set the IO tensor shapes.
        nvinfer1::Dims const shape{m_engine->getTensorShape(tensor_name)};
        // Print out dims.
        size_t tensor_size{1U};
        std::cout << "Tensor Dims: ";
        for (int32_t j{0}; j < shape.nbDims; ++j)
        {
            tensor_size *= shape.d[j];
            std::cout << shape.d[j] << " ";
        }
        std::cout << std::endl;

        if (tensor_name == std::string("input_image"))
        {
            tensor_size = m_input_channels * max_input_size * max_input_size;
        }
        else if (tensor_name == std::string("paded_image"))
        {
            tensor_size = m_input_channels * max_side_size * max_side_size;
        }

        size_t tensor_size_bytes{tensor_size * expected_dtype_byte_size};

        // Allocate host memory for the tensor.
        void* tensor_host_buffer{nullptr};
        CHECK_CUDA_ERROR(cudaMallocHost(&tensor_host_buffer, tensor_size_bytes));
        // Allocate device memory for the tensor.
        void* tensor_device_buffer{nullptr};
        CHECK_CUDA_ERROR(cudaMalloc(&tensor_device_buffer, tensor_size_bytes));

        bool status = m_context->setTensorAddress(tensor_name, tensor_device_buffer);

        if (!status)
            throw std::runtime_error("Failure in setTensorAddress");

        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input_tensor_host_buffers.push_back(tensor_host_buffer);
            input_tensor_device_buffers.push_back(tensor_device_buffer);
            input_tensor_shapes.push_back(shape);
            input_tensor_sizes.push_back(tensor_size);
            input_tensor_names.push_back(tensor_name);
        }
        else
        {
            output_tensor_host_buffers.push_back(tensor_host_buffer);
            output_tensor_device_buffers.push_back(tensor_device_buffer);
            output_tensor_shapes.push_back(shape);
            output_tensor_sizes.push_back(tensor_size);
            output_tensor_names.push_back(tensor_name);
        }
    }

    cv::Mat input_image = cv::imread(image_file_path);

    int input_width = input_image.cols;
    int input_height = input_image.rows;

    auto cv_depth_size = CV_32FC3;
    if (m_input_channels == 1)
    {
        cv::cvtColor(input_image, input_image, cv::COLOR_RGB2GRAY);
        cv_depth_size = CV_32FC1;
    }

    cv::Mat dest_image(cv::Size(input_image.cols, input_image.rows), cv_depth_size, (void*)(static_cast<float*>(input_tensor_host_buffers.at(0))));
    input_image.convertTo(dest_image, cv_depth_size);

    // Copy input data from host to device.
    size_t image_dtype_byte_size = 4;
    size_t const input_tensor_size_bytes{input_image.cols* input_image.rows * m_input_channels * image_dtype_byte_size};
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_tensor_device_buffers.at(0), input_tensor_host_buffers.at(0), input_tensor_size_bytes, cudaMemcpyHostToDevice, stream));

    char const* const input_image_tensor_name{input_tensor_names.at(0)};

    nvinfer1::Dims input_shape{m_engine->getTensorShape(input_image_tensor_name)};
    input_shape.d[1] = input_image.rows;
    input_shape.d[2] = input_image.cols;
    input_shape.d[3] = m_input_channels;
    m_context->setInputShape(input_image_tensor_name, input_shape);

    uint64_t pad_width = (std::ceil((double)input_width / patch_size) * patch_size) - input_width;
    uint64_t pad_height = (std::ceil((double)input_height / patch_size) * patch_size) - input_height;
    std::vector<uint64_t> pad_sizes{0,0,0,0, 0,pad_height,pad_width,0};

    size_t const pad_tensor_size_bytes{pad_sizes.size() * sizeof(uint64_t)};
    memcpy(input_tensor_host_buffers.at(1), pad_sizes.data(), pad_tensor_size_bytes);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(input_tensor_device_buffers.at(1), input_tensor_host_buffers.at(1), pad_tensor_size_bytes, cudaMemcpyHostToDevice, stream));
    char const* const pad_image_tensor_name{input_tensor_names.at(1)};
    nvinfer1::Dims pad_shape{m_engine->getTensorShape(pad_image_tensor_name)};
    m_context->setInputShape(pad_image_tensor_name, pad_shape);

    // Run inference a couple of times.
    bool const status{m_context->enqueueV3(stream)};
    if (!status)
    {
        std::cerr << "Failed to run inference." << std::endl;
        return EXIT_FAILURE;
    }

    // Synchronize.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy output data from device to host.
    size_t const output_tensor_size_bytes{output_tensor_sizes.at(0) * image_dtype_byte_size};
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output_tensor_host_buffers.at(0), output_tensor_device_buffers.at(0), output_tensor_size_bytes, cudaMemcpyDeviceToHost, stream));

    cv::Mat output_image(cv::Size(input_width + pad_width, input_height + pad_height), cv_depth_size, (char*)output_tensor_host_buffers.at(0));
    cv::imwrite("padded_output.jpg", output_image);

    std::cout << "Test finished";

    // Release resources.
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    for (size_t i{0U}; i < input_tensor_host_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(input_tensor_host_buffers.at(i)));
    }
    for (size_t i{0U}; i < input_tensor_device_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFree(input_tensor_device_buffers.at(i)));
    }
    for (size_t i{0U}; i < output_tensor_host_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFreeHost(output_tensor_host_buffers.at(i)));
    }
    for (size_t i{0U}; i < output_tensor_device_buffers.size(); ++i)
    {
        CHECK_CUDA_ERROR(cudaFree(output_tensor_device_buffers.at(i)));
    }

    return 0;
}
