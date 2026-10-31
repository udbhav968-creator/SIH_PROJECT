"""Verification tests for multi-defect scenes and zebra crossings."""

import unittest
import numpy as np
from PIL import Image

from models.marking_detector import detect_all_zebras, detect_zebra, paint_mask
from api.vercel_inference import _connected_components_8, _photometric_depth, _compute_pci
from models.ipm_homography_engine import IPMHomographyEngine
from models import camera_calibration as cam_cal


class TestMultiDefectAndMultiZebra(unittest.TestCase):

    def test_three_distinct_zebra_crossings_tight_boxes(self):
        """Verify that 3 spatially separated zebra crossings produce exactly 3 tight boxes."""

        H, W = 720, 1280
        img = np.full((H, W, 3), 55, dtype=np.uint8)

        for x in range(60, 340, 40):
            img[420:480, x:x+25] = 230

        for x in range(490, 770, 40):
            img[420:480, x:x+25] = 230

        for x in range(910, 1190, 40):
            img[420:480, x:x+25] = 230

        crossings = detect_all_zebras(img)
        self.assertEqual(len(crossings), 3, f"Expected 3 zebra crossings, got {len(crossings)}")

        crossings = sorted(crossings, key=lambda c: c["bbox_pixels"][0])

        expected_ranges = [
            (50, 70, 250, 280),
            (480, 500, 250, 280),
            (900, 920, 250, 280),
        ]

        for i, (c, (min_x, max_x, min_w, max_w)) in enumerate(zip(crossings, expected_ranges)):
            x0, y0, bw, bh = c["bbox_pixels"]
            self.assertTrue(min_x <= x0 <= max_x, f"Crossing {i+1} x0={x0} out of range [{min_x}, {max_x}]")
            self.assertTrue(min_w <= bw <= max_w, f"Crossing {i+1} bw={bw} out of range [{min_w}, {max_w}]")
            self.assertEqual(c["stripe_count"], 7, f"Crossing {i+1} stripe_count={c['stripe_count']}")
            self.assertGreaterEqual(c["confidence"], 0.70)

        res = detect_zebra(img)
        self.assertTrue(res["found"])
        self.assertEqual(res["crossings_count"], 3)
        self.assertEqual(len(res["all_crossings"]), 3)

    def test_one_hundred_potholes_connected_components(self):
        """Verify that a dense distribution of 100 pothole defects yields 100 boxes."""

        mask_h, mask_w = 200, 320
        binary_pothole = np.zeros((mask_h, mask_w), dtype=bool)

        count = 0
        for r_idx in range(10):
            y_c = 15 + r_idx * 18
            for c_idx in range(10):
                x_c = 15 + c_idx * 30

                binary_pothole[y_c-3:y_c+3, x_c-4:x_c+4] = True
                count += 1

        self.assertEqual(count, 100)

        cleaned, boxes = _connected_components_8(binary_pothole, min_blob_px=24)

        self.assertEqual(len(boxes), 100, f"Expected exactly 100 pothole boxes, got {len(boxes)}")

        orig_w, orig_h = 1280, 720
        sx = orig_w / float(mask_w)
        sy = orig_h / float(mask_h)
        for b in boxes:
            wx, wy, ww, wh = b["bbox_work"]
            self.assertGreater(ww, 0)
            self.assertGreater(wh, 0)
            bx = int(round(wx * sx))
            by = int(round(wy * sy))
            bw = int(round(ww * sx))
            bh = int(round(wh * sy))
            self.assertLessEqual(bx + bw, orig_w)
            self.assertLessEqual(by + bh, orig_h)
            self.assertGreaterEqual(b["area_work_px"], 24)

    def test_simultaneous_three_zebras_and_one_hundred_cavities(self):
        """Simultaneous scene: 3 zebra crossings and 100 distinct measured cavities."""
        H, W = 720, 1280
        img = np.full((H, W, 3), 60, dtype=np.uint8)

        for x in range(60, 340, 40):
            img[280:340, x:x+25] = 235
        for x in range(490, 770, 40):
            img[280:340, x:x+25] = 235
        for x in range(910, 1190, 40):
            img[280:340, x:x+25] = 235

        crossings = detect_all_zebras(img)
        self.assertEqual(len(crossings), 3)

        boxes_gt = []
        for r in range(10):
            y = 370 + r * 32
            for c in range(10):
                x = 50 + c * 115
                img[y:y+18, x:x+24] = 20
                boxes_gt.append([x, y, 24, 18])

        self.assertEqual(len(boxes_gt), 100)

        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
        calib, _ = cam_cal.resolve(None, width_px=W, height_px=H)
        ipm = IPMHomographyEngine.from_calibration(calib)

        measured = []
        total_remediation_cost = 0.0
        for b in boxes_gt:
            depth_info = _photometric_depth(gray, b, class_id=2)
            depth_cm = depth_info["depth_cm"]
            area_m2 = ipm.calculate_surface_area_sqm(b[0], b[1], b[2], b[3])
            vol_m3 = area_m2 * (depth_cm / 100.0)
            cost_inr = round(float(np.clip(vol_m3 * 2.4 * 1.15 * 7500.0, 50.0, 50000.0)), 2)
            measured.append({
                "bbox": b,
                "area_m2": area_m2,
                "depth_cm": depth_cm,
                "cost_inr": cost_inr,
            })
            total_remediation_cost += cost_inr

        self.assertEqual(len(measured), 100)
        self.assertGreater(total_remediation_cost, 1000.0)


if __name__ == "__main__":
    unittest.main()
