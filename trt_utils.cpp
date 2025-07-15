#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include "trt_utils.h"
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


//! \brief Create a BufferManager for handling buffer interactions with engine.
BufferManager::BufferManager(
    std::shared_ptr<nvinfer1::ICudaEngine> engine,
    std::map<std::string, std::vector<uint>> &max_dims,
    std::shared_ptr<nvinfer1::IExecutionContext> context
):
    m_engine(engine),
    m_max_dims(max_dims),
    m_context(context)
{
    nvinfer1::TensorFormat const expected_format{nvinfer1::TensorFormat::kLINEAR};

    int32_t const num_io_tensors{m_engine->getNbIOTensors()};
    LOG(INFO) << "Number of IO Tensors: " << num_io_tensors;

    for (int32_t i{0}; i < num_io_tensors; ++i)
    {
        char const* const tensor_name{m_engine->getIOTensorName(i)};
        LOG(INFO) << "Tensor name: " << tensor_name;
        nvinfer1::TensorIOMode const io_mode{m_engine->getTensorIOMode(tensor_name)};
        nvinfer1::DataType const d_type{m_engine->getTensorDataType(tensor_name)};

        nvinfer1::TensorFormat const format{m_engine->getTensorFormat(tensor_name)};
        if (format != expected_format)
        {
            LOG(ERROR) << "Invalid tensor format for tensor: " << tensor_name
                       << ". Expected: " << static_cast<int32_t>(expected_format)
                       << ", got: " << static_cast<int32_t>(format);
            throw std::runtime_error("Invalid tensor format");
        }

        nvinfer1::Dims shape{m_engine->getTensorShape(tensor_name)};
        size_t tensor_size{1U};
        for (int32_t j{0}; j < shape.nbDims; ++j)
        {
            if (shape.d[j] == -1)
            {
                if (std::find(m_max_dims[tensor_name].begin(), m_max_dims[tensor_name].end(), 0) != m_max_dims[tensor_name].end())
                {
                    LOG(ERROR) << "Dynamic tensor: " << tensor_name << " shape: " << shape << " not provided";
                    throw std::runtime_error("Invalid tensor shape");
                }

                if (j < static_cast<int32_t>(m_max_dims[tensor_name].size()))
                {
                    shape.d[j] = m_max_dims[tensor_name][j];
                }
                else
                {
                    LOG(ERROR) << "Invalid tensor shape: " << shape << ". Dimension " << j << " is dynamic and no size provided.";
                    throw std::runtime_error("Invalid tensor shape");
                }
            }

            tensor_size *= shape.d[j];
        }

        LOG(INFO) << "Tensor Dims: " << shape;

        size_t tensor_size_bytes{tensor_size * get_element_size(d_type)};

        std::unique_ptr<_managedBuffer> man_buf{new _managedBuffer()};
        man_buf->device_buffer = _deviceBuffer(tensor_size, d_type);
        man_buf->host_buffer = _hostBuffer(tensor_size, d_type);

        m_tensor_names.push_back(tensor_name);
        // m_device_bindings.emplace_back(manBuf->deviceBuffer.data());
        m_managed_buffers.emplace_back(std::move(man_buf));

        bool status = m_context->setTensorAddress(tensor_name, m_managed_buffers.back()->device_buffer.data());
        if (!status)
            throw std::runtime_error("Failure in setTensorAddress");

    }
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns the device buffer corresponding to tensor_name.
//!        Returns nullptr if no such tensor can be found.
void* BufferManager::get_device_buffer(const std::string& tensor_name) const
{
    return get_buffer(false, tensor_name);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns the host buffer corresponding to tensor_name.
//!        Returns nullptr if no such tensor can be found.
void* BufferManager::get_host_buffer(const std::string& tensor_name) const
{
    return get_buffer(true, tensor_name);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns the size of the host and device buffers that correspond to tensor_name.
size_t BufferManager::size(const std::string& tensor_name) const
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    int index = it_tensor - m_tensor_names.begin();
    return m_managed_buffers[index]->host_buffer.nb_bytes();
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of input host buffers to input device buffers synchronously.
// void BufferManager::copy_input_to_device()
// {
//     for (int i = 0; i < m_engine->getNbIOTensors(); i++)
//     {
//         void* dstPtr = m_managed_buffers[i]->device_buffer.data();
//         const void* srcPtr = m_managed_buffers[i]->host_buffer.data();
//         const size_t byteSize = m_managed_buffers[i]->host_buffer.nb_bytes();
//         if (m_engine->getTensorIOMode(m_engine->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT)
//         {
//             CHECK_CUDA_ERROR(cudaMemcpy(dstPtr, srcPtr, byteSize, cudaMemcpyHostToDevice));
//         }
//     }
// }
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of output device buffers to output host buffers synchronously.
// void BufferManager::copy_output_to_host()
// {
//     for (int i = 0; i < m_engine->getNbIOTensors(); i++)
//     {
//         void* dstPtr = m_managed_buffers[i]->host_buffer.data();
//         const void* srcPtr = m_managed_buffers[i]->device_buffer.data();
//         const size_t byteSize = m_managed_buffers[i]->host_buffer.nb_bytes();
//         if (m_engine->getTensorIOMode(m_engine->getIOTensorName(i)) != nvinfer1::TensorIOMode::kINPUT)
//         {
//             CHECK_CUDA_ERROR(cudaMemcpy(dstPtr, srcPtr, byteSize, cudaMemcpyDeviceToHost));
//         }
//     }
// }
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
void BufferManager::copy_input_to_device_async(const std::string& tensor_name, size_t buf_size, const cudaStream_t& stream)
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    int index = it_tensor - m_tensor_names.begin();

    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        m_managed_buffers[index]->device_buffer.data(),
        m_managed_buffers[index]->host_buffer.data(),
        buf_size,
        cudaMemcpyHostToDevice,
        stream
    ));
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
void BufferManager::copy_output_to_host_async(const cudaStream_t& stream)
{
    for (int i = 0; i < m_engine->getNbIOTensors(); i++)
    {
        if (m_engine->getTensorIOMode(m_engine->getIOTensorName(i)) != nvinfer1::TensorIOMode::kINPUT)
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                m_managed_buffers[i]->host_buffer.data(),
                m_managed_buffers[i]->device_buffer.data(),
                m_managed_buffers[i]->host_buffer.nb_bytes(),
                cudaMemcpyDeviceToHost,
                stream
            ));
        }
    }

    // Synchronize.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}
// --------------------------------------------------------------------------------------------------------------------------------

void* BufferManager::get_buffer(const bool isHost, const std::string& tensor_name) const
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    int index = it_tensor - m_tensor_names.begin();
    return (isHost ? m_managed_buffers[index]->host_buffer.data() : m_managed_buffers[index]->device_buffer.data());
}
// --------------------------------------------------------------------------------------------------------------------------------

uint32_t BufferManager::get_buffer_element_size(const std::string& tensor_name) const
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    return get_element_size(m_engine->getTensorDataType(tensor_name.c_str()));
};
// --------------------------------------------------------------------------------------------------------------------------------

void BufferManager::set_input_shape(const std::string& tensor_name, const std::vector<int> input_shape)
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    nvinfer1::Dims input_tensor_shape;
    input_tensor_shape.nbDims = input_shape.size();
    for (int i = 0; i < input_tensor_shape.nbDims; ++i)
    {
        input_tensor_shape.d[i] = input_shape[i];
    }

    m_context->setInputShape(tensor_name.c_str(), input_tensor_shape);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Return an ArrayND instance to access buffer data as array
cv::Mat BufferManager::get_mat(const std::string& tensor_name, uint batch_size)
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    nvinfer1::DataType type = m_engine->getTensorDataType(tensor_name.c_str());
    int cvtype;
    switch (type)
    {
        case nvinfer1::DataType::kINT32:
            cvtype = CV_32S;
            break;

        case nvinfer1::DataType::kFLOAT:
            cvtype = CV_32FC1;
            break;

        case nvinfer1::DataType::kHALF:
            cvtype = CV_16FC1;
            break;

        case nvinfer1::DataType::kBOOL:
            cvtype = CV_8U;
            break;

        case nvinfer1::DataType::kUINT8:
            cvtype = CV_8U;
            break;

        case nvinfer1::DataType::kINT8:
            cvtype = CV_8S;
            break;

        default:
            throw std::runtime_error("Invalid buffer type: " + std::to_string((int)type) + " for tensor: " + tensor_name);
    }

    auto dims = m_engine->getTensorShape(tensor_name.c_str());
    for (int32_t j{0}; j < dims.nbDims; ++j)
    {
        if (dims.d[j] == -1)
        {
            if (std::find(m_max_dims[tensor_name].begin(), m_max_dims[tensor_name].end(), 0) != m_max_dims[tensor_name].end())
            {
                LOG(ERROR) << "Dynamic tensor: " << tensor_name << " shape: " << dims << " not provided";
                throw std::runtime_error("Invalid tensor shape");
            }

            if (j < static_cast<int32_t>(m_max_dims[tensor_name].size()))
            {
                dims.d[j] = m_max_dims[tensor_name][j];
            }
        }
    }

    std::vector<int> dm;
    for (int d = 0; d < dims.nbDims; d++)
        dm.push_back(dims.d[d]);

    int index = it_tensor - m_tensor_names.begin();
    cv::Mat ret_mat(dm, cvtype, m_managed_buffers[index]->host_buffer.data());
    return std::move(ret_mat);
}
// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------

void check(cudaError_t err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        LOG(ERROR) << "CUDA Runtime Error at: " << file << ":" << line;
        LOG(ERROR) << cudaGetErrorString(err) << " " << func;
        std::exit(EXIT_FAILURE);
    }
}
// --------------------------------------------------------------------------------------------------------------------------------

void check_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        LOG(ERROR) << "CUDA Runtime Error at: " << file << ":" << line;
        LOG(ERROR) << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
// --------------------------------------------------------------------------------------------------------------------------------

json get_model_props(std::string dataset_id)
{
	std::string model_props_file = "/opt/eyeflow/data/models/"  + dataset_id + "_props.json";
    if (!fs::is_regular_file(model_props_file))
        throw std::invalid_argument("Model Props not found: " + model_props_file);

    std::ifstream fp;
    fp.open(model_props_file);
    if (!fp)
    {
        std::string err_msg = "Failed to open model props file: " + model_props_file + " - " + strerror(errno);
        LOG(ERROR) << err_msg;
        throw std::system_error(errno, std::system_category(), err_msg);
    }

    json ret_model_props = json::parse(fp);
    fp.close();

    for (auto prop : ret_model_props.items())
        LOG(INFO) << "Model prop: " << prop.key() << " - " << prop.value();

    return ret_model_props;
}
// --------------------------------------------------------------------------------------------------------------------------------
