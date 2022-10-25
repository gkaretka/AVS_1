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
    values_real = (float *)(aligned_alloc(64, height/2 * width * sizeof(float)));
    values_img = (float *)(aligned_alloc(64, height/2 * width * sizeof(float)));
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

    #pragma omp simd
    for (int j = 0; j < width; j++) {
        float v = x_start + j * dx;
        #pragma omp simd
        for (int i = 0; i < height/2; i++) {
            values_real[i*width+j] = v;
            data[i*width+j] = 100;
        }
    }

    #pragma omp simd
    for (int i = 0; i < height/2; i++) {
        float v = y_start + i * dy;
        #pragma omp simd
        for (int j = 0; j < width; j++) {
            values_img[i*width+j] = v;
        }
    }
}

int *LineMandelCalculator::calculateMandelbrot() {
    InitArray();

    const int _width = width;
    const int _height = height;
    const int _limit = limit;

    for (int i = 0; i < _height/2; i++) {
        for (int j = 0; j < _limit; ++j) {

            #pragma omp simd //simdlen(16)
            for (int k = 0; k < _width; k++) {
                float real = values_real[i*_width+k];
                float img = values_img[i*_width+k];
                float r2 = real * real;
                float i2 = img * img;

                values_img[i*_width+k] = (real == 0 || r2 + i2 > 4.0f) ? 0 : 2.0f * real * img + (y_start + i * dy);
                values_real[i*_width+k] = (real == 0 || r2 + i2 > 4.0f) ? 0 : r2 - i2 + (x_start + k * dx);
                
                data[i*_width+k] = (r2 + i2 > 4.0f) ? j : data[i*_width+k];
            }
        }
    }
    
    #pragma omp simd
    for (int i = _height/2; i < _height; i++) {
        #pragma omp simd
        for (int j = 0; j < _width; j++) {
            data[i*_width+j] = data[(height-i-1)*_width+j];
        }
    }

    return data;
}