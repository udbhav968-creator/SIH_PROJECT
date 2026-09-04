/*
 * ROAD-SHIELD Embedded Edge Neural Inference Header (C99 / C++ Compatible)
 * High-performance, zero-dependency embedded inference for MoRTH Patrol Vehicles.
 * Authority: MoRTH / NHAI SIH2026-MORTH-TRANS-018
 */
#ifndef ROAD_SHIELD_EDGE_INFERENCE_H
#define ROAD_SHIELD_EDGE_INFERENCE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline void edge_relu(float* vec, int len) {
    for (int i = 0; i < len; i++) {
        if (vec[i] < 0.0f) vec[i] = 0.0f;
    }
}

static inline void edge_softmax(const float* in, float* out, int len) {
    float max_val = in[0];
    for (int i = 1; i < len; i++) {
        if (in[i] > max_val) max_val = in[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        out[i] = expf(in[i] - max_val);
        sum += out[i];
    }
    float inv_sum = (sum > 1e-7f) ? (1.0f / sum) : 1.0f;
    for (int i = 0; i < len; i++) {
        out[i] *= inv_sum;
    }
}

static inline float morth_calculate_asphalt_tonnage(float area_m2, float depth_cm) {
    float vol_m3 = area_m2 * (depth_cm / 100.0f);
    return vol_m3 * 2.40f;
}

#ifdef __cplusplus
}
#endif

#endif // ROAD_SHIELD_EDGE_INFERENCE_H
