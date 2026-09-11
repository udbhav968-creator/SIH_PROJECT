/*
 * ROAD-SHIELD embedded formula library (C99 / C++ compatible).
 *
 * Ports the deterministic ASTM D6433 PCI deduct-value curve and the
 * pavement deterioration growth model to dependency-free C for
 * microcontroller / roadside-unit deployment.
 *
 * This header does NOT implement the scikit-learn SVM / RandomForest
 * classifiers used elsewhere in this project - real on-device inference
 * for those needs their fitted parameters (support vectors, tree splits)
 * exported separately alongside a small inference runtime, which is a
 * real but separate engineering task from this header.
 */
#ifndef ROAD_SHIELD_EDGE_INFERENCE_H
#define ROAD_SHIELD_EDGE_INFERENCE_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Saturating exponential deduct-value curve, ASTM D6433 style. */
static inline float rs_pci_deduct_value(float severity_cap, float density_pct) {
    return severity_cap * (1.0f - expf(-0.06f * density_pct));
}

/* Asphalt tonnage for a rectangular patch: area(m^2) * depth(m) * density(t/m^3). */
static inline float rs_asphalt_tonnage(float area_m2, float depth_cm, float density_t_per_m3) {
    float vol_m3 = area_m2 * (depth_cm / 100.0f);
    return vol_m3 * density_t_per_m3;
}

/* Exponential crack/pothole area growth over `days`, given a per-day rate k. */
static inline float rs_forecast_area(float init_area_m2, float k_per_day, float days) {
    return init_area_m2 * expf(k_per_day * days);
}

#ifdef __cplusplus
}
#endif

#endif /* ROAD_SHIELD_EDGE_INFERENCE_H */
