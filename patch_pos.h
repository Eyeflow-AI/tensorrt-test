#pragma once

#include <opencv2/opencv.hpp>

// --------------------------------------------------------------------------------------------------------------------------------

class PatchPos
{
public:
    PatchPos(uint patch_size, uint max_image_side, uint input_channels, const cv::Mat& input_frame, void* host_buffer, int host_type_size, int host_max_batch_size):
        m_patch_size(patch_size),
		m_patch_stride(patch_size / 2),
        m_max_image_side(max_image_side),
        m_input_channels(input_channels)
    {
        m_host_buffer = host_buffer;
        m_host_type_size = host_type_size;
        m_host_max_batch_size = host_max_batch_size;

        int frame_type;
        if (m_input_channels == 1)
        {
            if (input_frame.channels() == 3)
                cv::cvtColor(input_frame, m_frame, cv::COLOR_RGB2GRAY);
            else
                m_frame = input_frame;

            frame_type = CV_32FC1;
        }
        else if (m_input_channels == 3)
        {
            // throw std::runtime_error("Cannot work with 3 channels");
            if (input_frame.channels() == 1)
            	cv::cvtColor(input_frame, m_frame, cv::COLOR_GRAY2BGR);
            else
            	m_frame = input_frame;

            frame_type = CV_32FC3;
        }

        m_input_height = m_frame.rows;
        m_input_width = m_frame.cols;
        m_resize_scale = std::min((double)m_max_image_side / std::max(m_input_height, m_input_width), 1.0);
        if (m_resize_scale < 1)
        {
            m_input_height = ceil(m_input_height * m_resize_scale);
            m_input_width = ceil(m_input_width * m_resize_scale);

            cv::resize(m_frame, m_frame, cv::Size(m_input_width, m_input_height));
        }

        m_input_height = ceil((double)m_input_height / m_patch_stride) * m_patch_stride;
        m_input_width = ceil((double)m_input_width / m_patch_stride) * m_patch_stride;

        if ((m_input_height > m_frame.rows) || (m_input_width > m_frame.cols))
        {
            int down = (m_input_height - m_frame.rows);
            int right = (m_input_width - m_frame.cols);
            cv::copyMakeBorder(m_frame, m_frame, 0, down, 0, right, cv::BORDER_CONSTANT, cv::Scalar(0));
            if (!m_frame.isContinuous())
                m_frame = m_frame.clone();
        }

		m_num_patches_even_w = m_input_width / m_patch_size;
		m_num_patches_even_h = (m_input_height / m_patch_size);
		m_num_patches_even = m_num_patches_even_w * m_num_patches_even_h;
		m_num_patches_odd_w = (m_input_width - m_patch_stride) / m_patch_size;
		m_num_patches_odd_h = ((m_input_height - m_patch_stride) / m_patch_size);
		m_num_patches_total = m_num_patches_even + (m_num_patches_odd_w * m_num_patches_odd_h);

		m_patch_bytes = m_patch_size * m_patch_size * m_input_channels * m_host_type_size;
	};

	cv::Point get_image_pos(uint patch_num) const
	{
		if (patch_num < m_num_patches_even)
		{
			uint patch_h = patch_num / m_num_patches_even_w;
			uint patch_w = patch_num - (patch_h * m_num_patches_even_w);
			return cv::Point(patch_w * m_patch_size, patch_h * m_patch_size);
		}
		else
		{
            patch_num -= m_num_patches_even;
			uint patch_h = patch_num / m_num_patches_odd_w;
			uint patch_w = patch_num - (patch_h * m_num_patches_odd_w);
			return cv::Point(patch_w * m_patch_size + m_patch_stride, patch_h * m_patch_size + m_patch_stride);
		}
	};

	uint get_buffer_pos(uint patch_num) const
	{
		return ((patch_num % m_host_max_batch_size) * m_patch_bytes);
	};

    cv::Mat get_image_grid() const
    {
        cv::Mat grid(cv::Size(m_num_patches_total, 4), CV_32FC1);
        for (uint patch_num = 0; patch_num < m_num_patches_total; patch_num++)
        {
            if (patch_num < m_num_patches_even)
            {
                uint patch_h = patch_num / m_num_patches_even_w;
                uint patch_w = patch_num - (patch_h * m_num_patches_even_w);
                grid.at<float>(0, patch_num) = (float)(patch_w * m_patch_size);
                grid.at<float>(1, patch_num) = (float)(patch_h * m_patch_size);
                grid.at<float>(2, patch_num) = (float)(patch_w * m_patch_size);
                grid.at<float>(3, patch_num) = (float)(patch_h * m_patch_size);
            }
            else
            {
                uint patch_h = (patch_num - m_num_patches_even) / m_num_patches_odd_w;
                uint patch_w = (patch_num - m_num_patches_even) - (patch_h * m_num_patches_odd_w);
                grid.at<float>(0, patch_num) = (float)(patch_w * m_patch_size + m_patch_stride);
                grid.at<float>(1, patch_num) = (float)(patch_h * m_patch_size + m_patch_stride);
                grid.at<float>(2, patch_num) = (float)(patch_w * m_patch_size + m_patch_stride);
                grid.at<float>(3, patch_num) = (float)(patch_h * m_patch_size + m_patch_stride);
            }
        }

        return grid.t();
    };

    void copy_image_patch(uint patch_idx)
    {
        cv::Point img_pos = get_image_pos(patch_idx);
        uint buf_pos = get_buffer_pos(patch_idx);
        cv::Mat dest_patch(cv::Size(m_patch_size, m_patch_size), CV_32FC1, (void*)((uint8_t*)m_host_buffer + buf_pos));
        m_frame(cv::Range(img_pos.y, img_pos.y + m_patch_size), cv::Range(img_pos.x, img_pos.x + m_patch_size)).convertTo(dest_patch, CV_32FC1);
    }

    cv::Mat get_image_patch(uint patch_idx)
    {
        cv::Point img_pos = get_image_pos(patch_idx);
        uint buf_pos = get_buffer_pos(patch_idx);
        cv::Mat dest_patch(cv::Size(m_patch_size, m_patch_size), CV_32FC1, (void*)((uint8_t*)m_host_buffer + buf_pos));
        cv::Mat ret_frame;
        dest_patch.convertTo(ret_frame, CV_32FC1);
        return ret_frame;
    }

    void fill_slices_batch(uint start_patch, uint end_patch)
    {
        for (uint patch_idx = start_patch; patch_idx < end_patch; patch_idx++)
            copy_image_patch(patch_idx);
    }

    void print_grid(cv::Mat& grid)
    {
        for (int x = 0; x < grid.size[0]; x++)
        {
            std::stringstream box_line;
            for (int y = 0; y < 3; y++)
            // for (int y = 0; y < grid.size[1]; y++)
            {
                box_line << "[";
                std::stringstream box_add;
                for (int z = 0; z < grid.size[2]; z++)
                {
                    box_add << std::to_string(grid.at<float>(x, y, z));
                    if (z < grid.size[2] - 1)
                        box_add << ", ";
                }

                box_line << box_add.str() << "]";
                if (y < grid.size[1] - 1)
                    box_line << " | ";
            }
            std::cout << x << " - " << box_line.str() << std::endl;
        }
    }

	uint m_num_patches_total;
	uint m_num_patches_even_w;
	uint m_num_patches_even_h;
	uint m_num_patches_even;
	uint m_num_patches_odd_w;
	uint m_num_patches_odd_h;

    double m_resize_scale;
	uint m_input_width;
	uint m_input_height;

private:
    cv::Mat m_frame;
    void* m_host_buffer;
    int m_host_type_size;
    int m_host_max_batch_size;
	uint m_patch_size;
    uint m_max_image_side;
    uint m_input_channels;
	uint m_patch_stride;
	uint m_patch_bytes;
};
// --------------------------------------------------------------------------------------------------------------------------------
