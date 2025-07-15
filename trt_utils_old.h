#pragma once

#include <string>
#include <vector>
#include <memory>
#include <numeric>

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;
// --------------------------------------------------------------------------------------------------------------------------------

namespace trt_utils
{

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}
// --------------------------------------------------------------------------------------------------------------------------------

inline int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(&d.d[0], d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}
// --------------------------------------------------------------------------------------------------------------------------------

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}
// --------------------------------------------------------------------------------------------------------------------------------

//!
//! \brief  The _genericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class _genericBuffer
{
public:
    //! \brief Construct an empty buffer.
    _genericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : m_size(0)
        , m_capacity(0)
        , m_type(type)
        , m_buffer(nullptr)
    {
    };

    //! \brief Construct a buffer with the specified allocation size in bytes.
    _genericBuffer(size_t size, nvinfer1::DataType type)
        : m_size(size)
        , m_capacity(size)
        , m_type(type)
    {
        if (!m_alloc_fn(&m_buffer, this->nbBytes()))
        {
            throw std::bad_alloc();
        }
    };

    ~_genericBuffer()
    {
        m_free_fn(m_buffer);
    };

    _genericBuffer(_genericBuffer&& buf)
        : m_size(buf.m_size)
        , m_capacity(buf.m_capacity)
        , m_type(buf.m_type)
        , m_buffer(buf.m_buffer)
    {
        buf.m_size = 0;
        buf.m_capacity = 0;
        buf.m_type = nvinfer1::DataType::kFLOAT;
        buf.m_buffer = nullptr;
    };

    _genericBuffer& operator=(_genericBuffer&& buf)
    {
        if (this != &buf)
        {
            m_free_fn(m_buffer);
            m_size = buf.m_size;
            m_capacity = buf.m_capacity;
            m_type = buf.m_type;
            m_buffer = buf.m_buffer;
            // Reset buf.
            buf.m_size = 0;
            buf.m_capacity = 0;
            buf.m_buffer = nullptr;
        }
        return *this;
    };

    //! \brief Returns pointer to underlying array.
    void* data()
    {
        return m_buffer;
    };

    //! \brief Returns pointer to underlying array.
    const void* data() const
    {
        return m_buffer;
    };

    //! \brief Returns the size (in number of elements) of the buffer.
    size_t size() const
    {
        return m_size;
    };

    //! \brief Returns the size (in bytes) of the buffer.
    size_t nbBytes() const
    {
        return this->size() * getElementSize(m_type);
    };

    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    void resize(size_t new_size)
    {
        m_size = new_size;
        if (m_capacity < new_size)
        {
            m_free_fn(m_buffer);
            if (!m_alloc_fn(&m_buffer, this->nbBytes()))
            {
                throw std::bad_alloc{};
            }
            m_capacity = new_size;
        }
    };

    //! \brief Overload of resize that accepts Dims
    void resize(const nvinfer1::Dims& dims)
    {
        return this->resize(volume(dims));
    };

private:
    size_t m_size{0}, m_capacity{0};
    nvinfer1::DataType m_type;
    void* m_buffer;
    AllocFunc m_alloc_fn;
    FreeFunc m_free_fn;
};
// --------------------------------------------------------------------------------------------------------------------------------

class _deviceAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

class _deviceFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

class _hostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        cudaHostAlloc(ptr, size, cudaHostAllocDefault);
        return *ptr != nullptr;
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

class _hostFree
{
public:
    void operator()(void* ptr) const
    {
        cudaFreeHost(ptr);
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

using _deviceBuffer = _genericBuffer<_deviceAllocator, _deviceFree>;
using _hostBuffer = _genericBuffer<_hostAllocator, _hostFree>;

//! \brief  The _managedBuffer class groups together a pair of corresponding device and host buffers.
class _managedBuffer
{
public:
    _deviceBuffer deviceBuffer;
    _hostBuffer hostBuffer;
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

//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
class BufferManager
{
public:
    //! \brief Create a BufferManager for handling buffer interactions with engine.
    BufferManager(
        std::shared_ptr<nvinfer1::ICudaEngine> engine,
        const std::vector<uint> batch_size,
        const nvinfer1::IExecutionContext* context = nullptr
    );

    ~BufferManager() = default;

    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    std::vector<void*>& getDeviceBindings();

    //! \brief Returns a vector of device buffers.
    const std::vector<void*>& getDeviceBindings() const;

    //! \brief Returns the device buffer corresponding to tensor_name.
    //!        Returns nullptr if no such tensor can be found.
    void* getDeviceBuffer(const std::string& tensor_name) const;

    //! \brief Returns the host buffer corresponding to tensor_name.
    //!        Returns nullptr if no such tensor can be found.
    void* getHostBuffer(const std::string& tensor_name) const;

    //! \brief Returns the host buffer corresponding to tensor_name.
    //!        Returns nullptr if no such tensor can be found.
    uint32_t getBufferElementSize(const std::string& tensor_name) const
    {
        auto it_tensor = std::find(m_tensor_names.begin(), m_tensor_names.end(), tensor_name);
        if (it_tensor == m_tensor_names.end())
            throw std::runtime_error("Invalid tensor name: " + tensor_name);

        return getElementSize(m_engine->getTensorDataType(tensor_name.c_str()));
    };

    //! \brief Returns the size of the host and device buffers that correspond to tensor_name.
    size_t size(const std::string& tensor_name) const;

    //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
    //!        rowCount parameter controls how many elements are on each line.
    //!        A rowCount of 1 means that there is only 1 element on each line.
    template <typename T>
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0);
        T* typedBuf = static_cast<T*>(buf);
        size_t numItems = bufSize / sizeof(T);
        for (int i = 0; i < static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
                os << typedBuf[i] << std::endl;
            else if (rowCount == 1)
                os << typedBuf[i];
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
                os << typedBuf[i];
            else if (i % rowCount == rowCount - 1)
                os << " " << typedBuf[i] << std::endl;
            else
                os << " " << typedBuf[i];
        }
    }


    //! \brief Return an ArrayND instance to access buffer data as array
    cv::Mat getMat(const std::string& tensor_name, uint batch_size=1)
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
        if (dims.d[0] == -1)
            dims.d[0] = batch_size;

        std::vector<int> dm;
        for (int d = 0; d < dims.nbDims; d++)
            dm.push_back(dims.d[d]);

        int index = it_tensor - m_tensor_names.begin();
        cv::Mat ret_mat(dm, cvtype, m_managed_buffers[index]->hostBuffer.data());
        return std::move(ret_mat);
    }


    //! \brief Copy the contents of input host buffers to input device buffers synchronously.
    void copyInputToDevice();

    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    void copyOutputToHost();

    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    void copyInputToDeviceAsync(const cudaStream_t& stream = 0);

    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    void copyOutputToHostAsync(const cudaStream_t& stream = 0);

private:
    void* getBuffer(const bool isHost, const std::string& tensor_name) const;

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0);

    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::vector<uint> m_dims_size;
    std::vector<std::string> m_tensor_names;
    std::vector<std::unique_ptr<_managedBuffer>> m_managed_buffers;
    std::vector<void*> m_device_bindings;
}; // class BufferManager
// --------------------------------------------------------------------------------------------------------------------------------


//! Class which manages logging of TensorRT
//! - Debugging messages with an associated severity (info, warning, error, or internal error/fatal)
class Logger : public nvinfer1::ILogger
{
public:
    static Logger *get_instance();

    nvinfer1::ILogger& getTRTLogger() noexcept;

    //! \brief Implementation of the nvinfer1::ILogger::log() virtual method
    void log(Severity severity, const char* msg) noexcept override;

    //! \brief Method for controlling the verbosity of logging output
    void setReportableSeverity(Severity severity) noexcept;

    Severity getReportableSeverity() const;

private:
    explicit Logger();

    Severity mReportableSeverity;
}; // class Logger
// --------------------------------------------------------------------------------------------------------------------------------


template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;
// --------------------------------------------------------------------------------------------------------------------------------

json get_model_props(std::string dataset_id);
// --------------------------------------------------------------------------------------------------------------------------------

} // namespace trt_utils
