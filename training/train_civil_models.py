"""
ROAD-SHIELD Civil Engineering Models Training Entrypoint
========================================================

Delegates directly to `training.train_civil_real` so that all civil models
(ASTM D6433-20 PCI, IRC:37-2018 Deterioration, and IRC:SP:83 / IRC:82 Depth)
are always trained on real highway photographs and human-drawn COCO polygon
annotations (`datasets/incoming/cracks_potholes_dnit/coco.json`), never on
synthetic random grids.
"""

from training.train_civil_real import (
    ASTM_CURVES,
    ASTM_DENSITY_GRID,
    SEV_MAP,
    astm_cdv,
    interp_astm_deduct,
    main,
    train_depth_model_real as train_depth_model,
    train_deterioration_model_real as train_deterioration_model,
    train_pci_model_real as train_pci_model,
)

if __name__ == "__main__":
    main()
