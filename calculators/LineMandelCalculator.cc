/**
 * @file LineMandelCalculator.cc
 * @author Gregor Karetka <xkaret00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization
 * over lines
 * @date DATE
 */
#include "LineMandelCalculator.h"

#include <stdlib.h>

#include <algorithm>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit)
    : BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator") {
    data = (int *)(aligned_alloc(64, height * width * sizeof(int)));
    values_real = (float *)(aligned_alloc(64, width * sizeof(float)));
    values_img = (float *)(aligned_alloc(64, width * sizeof(float)));
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    free(values_real);
    free(values_img);

    data = NULL;
    values_img = NULL;
    values_real = NULL;
}

inline void LineMandelCalculator::InitArray() {
    // init partial calculation array

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

    for (int i = 0; i < _height / 2; i++) {
        float v = _y_start + i * _dy;

#pragma omp simd simdlen(64) aligned(_data, _values_img, _values_real : 64)
        for (int j = 0; j < _width; j++) {
            _values_real[i * _width + j] = _x_start + j * _dx;
            _values_img[i * _width + j] = v;
        }
    }
}

int *LineMandelCalculator::calculateMandelbrot() {
    // InitArray();

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

    for (int i = 0; i < _height / 2; i++) {
        int done_counter = 0;
        const float _ysidy = _y_start + i * _dy;
        const int _width_pointer = i * _width;

#pragma omp simd simdlen(64) aligned(_data, _values_img, _values_real : 64)
        for (int m = 0; m < _width; m++) {
            _values_real[m] = _x_start + m * _dx;
            _values_img[m] = _ysidy;
        }
        for (int j = 1; done_counter < _width && j <= _limit; j++) {
            done_counter = 0;
#pragma omp simd simdlen(64) reduction(+ : done_counter) aligned(_data, _values_img, _values_real : 64)
            for (int k = 0; k < _width; k++) {
                float real = _values_real[k];
                float img = _values_img[k];
                float r2 = real * real;
                float i2 = img * img;
                _data[_width_pointer + k] = (r2 + i2 < 4.0f) ? j : _data[_width_pointer + k];

                _values_img[k] = 2.0f * real * img + _ysidy;
                _values_real[k] = r2 - i2 + (_x_start + k * _dx);

                if (r2 + i2 >= 4.0f) {
                    done_counter += 1;
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