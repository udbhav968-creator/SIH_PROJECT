"""
Edge / embedded model-export utility.

The real trained models in this project are scikit-learn pipelines
(StandardScaler -> PCA -> SVC for vision, StandardScaler -> RandomForest for
IMU shock) saved with joblib - not the hand-rolled NumPy weight arrays the
old version of this exporter looked for (npz files with keys like "w_cls",
"conv_w", "W_q" that no longer exist anywhere in this project, since there
never was a real CNN-Transformer to produce them).

This rewrite does two honest things instead:
1. Introspects the actual fitted scikit-learn pipelines and writes a real
   JSON description of their structure (kernel, support-vector counts,
   number of trees, feature counts, class names) - not a fabricated
   "CNN-Transformer hybrid" architecture.
2. Hand-ports the deterministic, formula-based engines (the ASTM D6433 PCI
   deduct-value curve and the pavement-deterioration growth model) to a
   dependency-free C header, since those are genuinely just arithmetic and
   cheap to run on a microcontroller. The SVM / RandomForest models are
   NOT exported to C here: real on-device inference for those needs their
   fitted parameters (support vectors, tree splits) bundled with a small
   runtime, which is a real but separate engineering task from generating
   a single static header.
"""

import os
import json
import time

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


class EdgeModelExporter:
    """Exports the project's real trained pipelines and formula-based engines for embedded use."""

    def __init__(self, checkpoints_dir):
        self.ckpt_dir = checkpoints_dir

    @staticmethod
    def _describe_sklearn_pipeline(pipeline):
        steps = []
        for name, step in pipeline.named_steps.items():
            cls = type(step).__name__
            info = {"step": name, "type": cls}
            if cls == "PCA":
                info["n_components"] = int(step.n_components_)
                info["explained_variance_ratio_sum"] = round(float(step.explained_variance_ratio_.sum()), 4)
            elif cls == "SVC":
                info["kernel"] = step.kernel
                info["n_support_vectors_per_class"] = [int(x) for x in step.n_support_]
                info["C"] = step.C
                info["gamma"] = str(step.gamma)
            elif cls == "RandomForestClassifier":
                info["n_estimators"] = int(step.n_estimators)
                info["max_depth"] = step.max_depth
                info["n_features_in"] = int(step.n_features_in_)
            steps.append(info)
        return steps

    def export_all_to_open_spec(self, output_dir=None):
        if output_dir is None:
            output_dir = self.ckpt_dir
        os.makedirs(output_dir, exist_ok=True)

        specs = {}

        if joblib is not None:
            vis_path = os.path.join(self.ckpt_dir, "vision_distress_model.joblib")
            if os.path.exists(vis_path):
                from models.vision_distress_net import VisionDistressNet

                model = VisionDistressNet(model_path=vis_path)
                if model.is_ready:
                    specs["Model_M1_VisionDistressNet"] = {
                        "architecture": "scikit-learn Pipeline: StandardScaler -> PCA -> SVC(RBF)",
                        "class_names": model.CLASS_NAMES,
                        "pipeline_steps": self._describe_sklearn_pipeline(model.pipeline),
                        "note": "Trained on real photos, split at the source-photo level before augmentation; "
                        "see checkpoints/vision_distress_report.json for honest held-out accuracy.",
                    }

            imu_path = os.path.join(self.ckpt_dir, "imu_shock_model.joblib")
            if os.path.exists(imu_path):
                from models.imu_shock_classifier import IMUShockClassifier

                model = IMUShockClassifier(model_path=imu_path)
                if model.is_ready:
                    specs["Model_M4_IMUShockClassifier"] = {
                        "architecture": "scikit-learn Pipeline: StandardScaler -> RandomForestClassifier",
                        "pipeline_steps": self._describe_sklearn_pipeline(model.pipeline),
                        "note": "See checkpoints/imu_shock_report.json for honest held-out accuracy.",
                    }

        # Deterministic formula-based engines - genuinely embeddable, no weights to export.
        from models.pci_regressor_net import PavementConditionIndexEngine

        specs["Model_PCI_ASTM_D6433"] = {
            "architecture": "Deterministic ASTM D6433 deduct-value formula (no trained weights)",
            "severity_caps": PavementConditionIndexEngine.SEVERITY_CAP,
        }

        from models.pavement_deterioration_forecaster import PavementDeteriorationForecaster

        specs["Model_Deterioration_Forecaster"] = {
            "architecture": "Deterministic exponential area-growth model (no trained weights)",
            "k_base_per_day": PavementDeteriorationForecaster.K_BASE,
            "k_per_1000_esal_per_day": PavementDeteriorationForecaster.K_PER_1000_ESAL,
            "k_per_100mm_rain_per_day": PavementDeteriorationForecaster.K_PER_100MM_RAIN,
        }

        spec_file = os.path.join(output_dir, "road_shield_open_model_spec.json")
        with open(spec_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "format": "ROAD-SHIELD Open Model Specification v2.0",
                    "generated_timestamp_utc": int(time.time()),
                    "note": "v2.0 replaces the earlier fabricated 'Open Neural Network' spec (a dump of an "
                    "untrained numpy CNN/transformer's random weights) with an honest description of the "
                    "real scikit-learn pipelines and deterministic formulas actually used in this project.",
                    "models": specs,
                },
                fh,
                indent=2,
            )

        header_file = self.generate_c_header(specs, output_dir)

        return {
            "spec_json_path": spec_file,
            "c_header_path": header_file,
            "models_exported": list(specs.keys()),
        }

    def generate_c_header(self, specs, output_dir):
        """
        Generates a dependency-free C header for the parts of this project
        that are genuinely cheap to hand-port: the ASTM D6433 PCI
        deduct-value curve, asphalt tonnage arithmetic, and the pavement
        deterioration growth model. Deliberately does not claim to
        implement the SVM / RandomForest classifiers in C.
        """
        header_path = os.path.join(output_dir, "road_shield_edge_inference.h")
        c_code = (
            "/*\n"
            " * ROAD-SHIELD embedded formula library (C99 / C++ compatible).\n"
            " *\n"
            " * Ports the deterministic ASTM D6433 PCI deduct-value curve and the\n"
            " * pavement deterioration growth model to dependency-free C for\n"
            " * microcontroller / roadside-unit deployment.\n"
            " *\n"
            " * This header does NOT implement the scikit-learn SVM / RandomForest\n"
            " * classifiers used elsewhere in this project - real on-device inference\n"
            " * for those needs their fitted parameters (support vectors, tree splits)\n"
            " * exported separately alongside a small inference runtime, which is a\n"
            " * real but separate engineering task from this header.\n"
            " */\n"
            "#ifndef ROAD_SHIELD_EDGE_INFERENCE_H\n"
            "#define ROAD_SHIELD_EDGE_INFERENCE_H\n\n"
            "#include <math.h>\n\n"
            "#ifdef __cplusplus\n"
            "extern \"C\" {\n"
            "#endif\n\n"
            "/* Saturating exponential deduct-value curve, ASTM D6433 style. */\n"
            "static inline float rs_pci_deduct_value(float severity_cap, float density_pct) {\n"
            "    return severity_cap * (1.0f - expf(-0.06f * density_pct));\n"
            "}\n\n"
            "/* Asphalt tonnage for a rectangular patch: area(m^2) * depth(m) * density(t/m^3). */\n"
            "static inline float rs_asphalt_tonnage(float area_m2, float depth_cm, float density_t_per_m3) {\n"
            "    float vol_m3 = area_m2 * (depth_cm / 100.0f);\n"
            "    return vol_m3 * density_t_per_m3;\n"
            "}\n\n"
            "/* Exponential crack/pothole area growth over `days`, given a per-day rate k. */\n"
            "static inline float rs_forecast_area(float init_area_m2, float k_per_day, float days) {\n"
            "    return init_area_m2 * expf(k_per_day * days);\n"
            "}\n\n"
            "#ifdef __cplusplus\n"
            "}\n"
            "#endif\n\n"
            "#endif /* ROAD_SHIELD_EDGE_INFERENCE_H */\n"
        )
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(c_code)
        return header_path


if __name__ == "__main__":
    ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    exporter = EdgeModelExporter(os.path.join(ENGINE_ROOT, "checkpoints"))
    res = exporter.export_all_to_open_spec()
    print("Export complete:")
    print("  Open model spec:", res["spec_json_path"])
    print("  C header:", res["c_header_path"])
    print("  Models exported:", res["models_exported"])
