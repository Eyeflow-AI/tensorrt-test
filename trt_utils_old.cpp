#include <glog/logging.h>

#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include "trt_utils_old.h"
// --------------------------------------------------------------------------------------------------------------------------------

namespace trt_utils
{

//! \brief Create a BufferManager for handling buffer interactions with engine.
BufferManager::BufferManager(
    std::shared_ptr<nvinfer1::ICudaEngine> engine,
    const std::vector<uint> dims_size,
    const nvinfer1::IExecutionContext* context
)
    : m_engine(engine)
    , m_dims_size(dims_size)
{
    // Create host and device buffers
    for (int i = 0; i < m_engine->getNbIOTensors(); i++)
    {
        const char* tensor_name = m_engine->getIOTensorName(i);
        auto dims = context ? context->getTensorShape(tensor_name) : m_engine->getTensorShape(tensor_name);
        nvinfer1::DataType type = m_engine->getTensorDataType(tensor_name);
        int vec_dim = m_engine->getTensorVectorizedDim(tensor_name);

        for (int d = 0; d < dims.nbDims; d++)
            if (dims.d[d] == -1)
                dims.d[d] = dims_size[d];

        size_t vol = volume(dims);
        std::unique_ptr<_managedBuffer> manBuf{new _managedBuffer()};
        manBuf->deviceBuffer = _deviceBuffer(vol, type);
        manBuf->hostBuffer = _hostBuffer(vol, type);

        m_tensor_names.push_back(tensor_name);
        m_device_bindings.emplace_back(manBuf->deviceBuffer.data());
        m_managed_buffers.emplace_back(std::move(manBuf));
    }
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns a vector of device buffers that you can use directly as
//!        bindings for the execute and enqueue methods of IExecutionContext.
std::vector<void*>& BufferManager::getDeviceBindings()
{
    return m_device_bindings;
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns a vector of device buffers.
const std::vector<void*>& BufferManager::getDeviceBindings() const
{
    return m_device_bindings;
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns the device buffer corresponding to tensor_name.
//!        Returns nullptr if no such tensor can be found.
void* BufferManager::getDeviceBuffer(const std::string& tensor_name) const
{
    return getBuffer(false, tensor_name);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns the host buffer corresponding to tensor_name.
//!        Returns nullptr if no such tensor can be found.
void* BufferManager::getHostBuffer(const std::string& tensor_name) const
{
    return getBuffer(true, tensor_name);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Returns the size of the host and device buffers that correspond to tensor_name.
size_t BufferManager::size(const std::string& tensor_name) const
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    int index = it_tensor - m_tensor_names.begin();
    return m_managed_buffers[index]->hostBuffer.nbBytes();
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of input host buffers to input device buffers synchronously.
void BufferManager::copyInputToDevice()
{
    memcpyBuffers(true, false, false);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of output device buffers to output host buffers synchronously.
void BufferManager::copyOutputToHost()
{
    memcpyBuffers(false, true, false);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of input host buffers to input device buffers asynchronously.
void BufferManager::copyInputToDeviceAsync(const cudaStream_t& stream)
{
    memcpyBuffers(true, false, true, stream);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Copy the contents of output device buffers to output host buffers asynchronously.
void BufferManager::copyOutputToHostAsync(const cudaStream_t& stream)
{
    memcpyBuffers(false, true, true, stream);
}
// --------------------------------------------------------------------------------------------------------------------------------

void* BufferManager::getBuffer(const bool isHost, const std::string& tensor_name) const
{
    auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
    if (it_tensor == m_tensor_names.end())
        throw std::runtime_error("Invalid tensor name: " + tensor_name);

    int index = it_tensor - m_tensor_names.begin();
    return (isHost ? m_managed_buffers[index]->hostBuffer.data() : m_managed_buffers[index]->deviceBuffer.data());
}
// --------------------------------------------------------------------------------------------------------------------------------

void BufferManager::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
    for (int i = 0; i < m_engine->getNbIOTensors(); i++)
    {
        void* dstPtr = deviceToHost ? m_managed_buffers[i]->hostBuffer.data() : m_managed_buffers[i]->deviceBuffer.data();
        const void* srcPtr = deviceToHost ? m_managed_buffers[i]->deviceBuffer.data() : m_managed_buffers[i]->hostBuffer.data();
        const size_t byteSize = m_managed_buffers[i]->hostBuffer.nbBytes();
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        bool input_tensor = (m_engine->getTensorIOMode(m_engine->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT);
        if ((copyInput && input_tensor) || (!copyInput && !input_tensor))
        {
            if (async)
            {
                auto ret = (cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                if (ret != 0)
                    throw std::runtime_error("Cuda failure: " + std::to_string(ret));
            }
            else
            {
                auto ret = (cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
                if (ret != 0)
                    throw std::runtime_error("Cuda failure: " + std::to_string(ret));
            }
        }
    }
}
// --------------------------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------------------------

// TRT Logger

static Logger *_singleton_logger = nullptr;

Logger* Logger::get_instance()
{
    /**
     * This is a safer way to create an instance. instance = new Singleton is
     * dangeruous in case two instance threads wants to access at the same time
     */
    if (_singleton_logger == nullptr)
    {
        _singleton_logger = new Logger();
    }
    return _singleton_logger;
}
// --------------------------------------------------------------------------------------------------------------------------------

Logger::Logger()
{
    mReportableSeverity = Severity::kINFO;
    // mReportableSeverity = Severity::kWARNING;
}
// --------------------------------------------------------------------------------------------------------------------------------

nvinfer1::ILogger& Logger::getTRTLogger() noexcept
{
    return *this;
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Implementation of the nvinfer1::ILogger::log() virtual method
void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    // suppress info-level messages
    if (severity <= mReportableSeverity)
        LOG(INFO) << "[TRT] " << std::string(msg);
}
// --------------------------------------------------------------------------------------------------------------------------------

//! \brief Method for controlling the verbosity of logging output
void Logger::setReportableSeverity(Severity severity) noexcept
{
    mReportableSeverity = severity;
}
// --------------------------------------------------------------------------------------------------------------------------------

nvinfer1::ILogger::Severity Logger::getReportableSeverity() const
{
    return mReportableSeverity;
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

} // namespace trt_utils
