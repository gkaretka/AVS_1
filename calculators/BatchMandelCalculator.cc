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
#pragma omp simd simdlen(16)
        for (int j = 0; j < _width; j++) {
            values_real[i * _width + j] = _x_start + j * _dx;
            data[i * _width + j] = _limit;
        }
    }
    for (int i = 0; i < _height / 2; i++) {
        float v = _y_start + i * _dy;
#pragma omp simd simdlen(16)
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

    for (int i = 0; i < _height / 2; i++) {
        for (int j = 0; j < _limit; ++j) {

#pragma omp simd simdlen(16)
            for (int k = 0; k < _width; k++) {
                float real = values_real[i * _width + k];
                float img = values_img[i * _width + k];
                float r2 = real * real;
                float i2 = img * img;

                data[i * _width + k] = (r2 + i2 > 4.0f) ? j : data[i * _width + k];
                values_img[i * _width + k] =
                    (real == 0 || r2 + i2 > 4.0f) ? 0 : 2.0f * real * img + (_y_start + i * _dy);
                values_real[i * _width + k] = (real == 0 || r2 + i2 > 4.0f) ? 0 : r2 - i2 + (_x_start + k * _dx);
            }
        }
    }

    for (int i = _height / 2; i < _height; i++) {
#pragma omp simd simdlen(16)
        for (int j = 0; j < _width; j++) {
            data[i * _width + j] = data[(height - i - 1) * _width + j];
        }
    }

    return data;
    return NULL;
}