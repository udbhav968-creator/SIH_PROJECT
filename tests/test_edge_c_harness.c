/*
 * Standalone verification harness for ROAD-SHIELD embedded C inference library.
 * Tests C99 compatibility, ASTM D6433 PCI deduct computation, and tonnage formula.
 */

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "../checkpoints/road_shield_edge_inference.h"

int main(void) {
    printf("[EDGE C HARNESS] Testing road_shield_edge_inference.h with gcc C99...\n");

    /* 1. ASTM D6433 deduct curve check */
    float deduct_zero = rs_pci_deduct_value(100.0f, 0.0f);
    assert(fabsf(deduct_zero) < 1e-5f);

    float deduct_typical = rs_pci_deduct_value(65.0f, 15.0f);
    assert(deduct_typical > 35.0f && deduct_typical < 45.0f);
    printf("  [PASS] rs_pci_deduct_value(65.0, 15.0) = %.2f pts\n", deduct_typical);

    /* 2. Asphalt tonnage formula check */
    float tonnage = rs_asphalt_tonnage(1.8f, 6.5f, 2.40f);
    /* 1.8 * 0.065 * 2.40 = 0.2808 tonnes */
    assert(fabsf(tonnage - 0.2808f) < 1e-4f);
    printf("  [PASS] rs_asphalt_tonnage(1.8m^2, 6.5cm, 2.40 t/m^3) = %.4f tonnes\n", tonnage);

    /* 3. Deterioration forecast check */
    float area_future = rs_forecast_area(1.0f, 0.005f, 180.0f);
    assert(area_future > 2.0f && area_future < 3.0f);
    printf("  [PASS] rs_forecast_area(1.0m^2, k=0.005, 180d) = %.3f m^2\n", area_future);

    printf("[EDGE C HARNESS] ALL C99 EMBEDDED FORMULA TESTS PASSED (0 ERRORS)\n");
    return 0;
}
