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

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line);
#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line);
// --------------------------------------------------------------------------------------------------------------------------------

inline uint32_t get_element_size(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
        case nvinfer1::DataType::kINT64: return 8;
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
    _genericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT):
        m_size(0),
        m_capacity(0),
        m_type(type),
        m_buffer(nullptr)
    {};

    //! \brief Construct a buffer with the specified allocation size in bytes.
    _genericBuffer(size_t size, nvinfer1::DataType type):
        m_size(size),
        m_capacity(size),
        m_type(type)
    {
        if (!m_alloc_fn(&m_buffer, this->nb_bytes()))
        {
            throw std::bad_alloc();
        }
    };

    ~_genericBuffer()
    {
        m_free_fn(m_buffer);
    };

    _genericBuffer(_genericBuffer&& buf):
        m_size(buf.m_size),
        m_capacity(buf.m_capacity),
        m_type(buf.m_type),
        m_buffer(buf.m_buffer)
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
    size_t nb_bytes() const
    {
        return m_size * get_element_size(m_type);
    };

private:
    nvinfer1::Dims m_tensor_dims;
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
        CHECK_CUDA_ERROR(cudaMalloc(ptr, size));
        return *ptr != nullptr;
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

class _deviceFree
{
public:
    void operator()(void* ptr) const
    {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

class _hostAllocator
{
public:
    bool operator()(void** ptr, size_t size) const
    {
        CHECK_CUDA_ERROR(cudaMallocHost(ptr, size));
        return *ptr != nullptr;
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

class _hostFree
{
public:
    void operator()(void* ptr) const
    {
        CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    }
};
// --------------------------------------------------------------------------------------------------------------------------------

using _deviceBuffer = _genericBuffer<_deviceAllocator, _deviceFree>;
using _hostBuffer = _genericBuffer<_hostAllocator, _hostFree>;

//! \brief  The _managedBuffer class groups together a pair of corresponding device and host buffers.
class _managedBuffer
{
public:
    _deviceBuffer device_buffer;
    _hostBuffer host_buffer;
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
        std::map<std::string, std::vector<uint>> &max_dims,
        std::shared_ptr<nvinfer1::IExecutionContext> context
    );

    ~BufferManager() = default;

    //! \brief Returns the device buffer corresponding to tensor_name.
    //!        Returns nullptr if no such tensor can be found.
    void* get_device_buffer(const std::string& tensor_name) const;

    //! \brief Returns the host buffer corresponding to tensor_name.
    //!        Returns nullptr if no such tensor can be found.
    void* get_host_buffer(const std::string& tensor_name) const;

    //! \brief Returns the host buffer corresponding to tensor_name.
    //!        Returns nullptr if no such tensor can be found.
    uint32_t get_buffer_element_size(const std::string& tensor_name) const;

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

    cv::Mat get_mat(const std::string& tensor_name, uint batch_size=1);

    //! \brief Copy the contents of input host buffers to input device buffers synchronously.
    // void copy_input_to_device();

    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    // void copy_output_to_host();

    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    void copy_input_to_device_async(const std::string& tensor_name, size_t buf_size, const cudaStream_t& stream);

    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    void copy_output_to_host_async(const cudaStream_t& stream);

    void set_input_shape(const std::string& tensor_name, const std::vector<int> input_shape);

private:
    void* get_buffer(const bool isHost, const std::string& tensor_name) const;
    // void mem_cpy_buffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0);

    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context;
    std::vector<std::string> m_tensor_names;
    std::vector<std::unique_ptr<_managedBuffer>> m_managed_buffers;
    std::map<std::string, std::vector<uint>> m_max_dims;
}; // class BufferManager
// --------------------------------------------------------------------------------------------------------------------------------

class CustomLogger : public nvinfer1::ILogger
{
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= nvinfer1::ILogger::Severity::kINFO)
        {
            LOG(INFO) << "[TRT] " << std::string(msg);
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

// template <typename T>
// using InferUniquePtr = std::unique_ptr<T, InferDeleter>;
// --------------------------------------------------------------------------------------------------------------------------------

json get_model_props(std::string dataset_id);
// --------------------------------------------------------------------------------------------------------------------------------
