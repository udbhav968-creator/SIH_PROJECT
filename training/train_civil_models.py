"""Train ASTM D6433 and HDM-4 regression models."""

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
