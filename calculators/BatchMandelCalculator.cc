/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
    data = (int *)(aligned_alloc(64, height * width * sizeof(int)));
    values_real = (float *)(aligned_alloc(64, height / 2 * width * sizeof(float)));
    values_img = (float *)(aligned_alloc(64, height / 2 * width * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
    free(data);
    free(values_real);
    free(values_img);

    data = NULL;
    values_img = NULL;
    values_real = NULL;
}

inline void BatchMandelCalculator::InitArray() {
    // init partial calculation array

    const int _width = width;
    const int _height = height;
    const int _limit = limit;
    const float _x_start = x_start;
    const float _y_start = y_start;
    const float _dx = dx;
    const float _dy = dy;

    for (int i = 0; i < _height / 2; i++) {
#pragma omp simd simdlen(64)
        for (int j = 0; j < _width; j++) {
            values_real[i * _width + j] = _x_start + j * _dx;
            data[i * _width + j] = _limit;
        }
    }
    for (int i = 0; i < _height / 2; i++) {
        float v = _y_start + i * _dy;
#pragma omp simd simdlen(64)
        for (int j = 0; j < _width; j++) {
            values_img[i * _width + j] = v;
        }
    }
}

int * BatchMandelCalculator::calculateMandelbrot () {
    InitArray();

    const int _width = width;
    const int _height = height;
    const int _limit = limit;
    const float _x_start = x_start;
    const float _y_start = y_start;
    const float _dx = dx;
    const float _dy = dy;
    float *_values_img = values_img;
    float *_values_real = values_real;
    int *_data = data;

    const int batch_size_x = 256;

    // block for width
    for (int batchk = 0; batchk < _width / batch_size_x; batchk++) {
        const int batch_k_start = batchk * batch_size_x;

        // block for height
        for (int i = 0; i < (_height / 2); i++) {
            const float _ysidy = _y_start + i * _dy;
            const int _width_pointer = i * _width + batch_k_start;
            int done_counter = 0;

            // limit check
            for (int j = 0; done_counter < batch_size_x && j < _limit; ++j) {
                // for width in block
#pragma omp simd simdlen(64) reduction(+ : done_counter) aligned(_data, _values_img, _values_real : 64)
                for (int k = 0; k < batch_size_x; k++) {
                    float real = _values_real[_width_pointer + k];
                    float img = _values_img[_width_pointer + k];
                    float r2 = real * real;
                    float i2 = img * img;

                    _data[_width_pointer + k] = (r2 + i2 > 4.0f) ? j : _data[_width_pointer + k];

                    _values_img[_width_pointer + k] = (real == 0 || r2 + i2 > 4.0f) ? 0 : 2.0f * real * img + _ysidy;
                    _values_real[_width_pointer + k] =
                        (real == 0 || r2 + i2 > 4.0f) ? 0 : r2 - i2 + (_x_start + (k + batch_k_start) * _dx);
                    if (r2 + i2 > 4.0f) {
                        done_counter += 1;
                    }
                }
            }
        }
    }

    for (int i = _height / 2; i < _height; i++) {
#pragma omp simd simdlen(64)
        for (int j = 0; j < _width; j++) {
            _data[i * _width + j] = _data[(_height - i - 1) * _width + j];
        }
    }
    return data;
}