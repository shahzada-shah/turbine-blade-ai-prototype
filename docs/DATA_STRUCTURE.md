# Data Structure Guide

## Expected Directory Structure

For training and testing the BladeGuard model, organize your blade images in the following structure:

```
data/
├── train/
│   ├── healthy/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── minor_damage/
│   │   ├── image1.jpg
│   │   └── ...
│   └── severe_damage/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── healthy/
│   ├── minor_damage/
│   └── severe_damage/
└── test/
    ├── healthy/
    ├── minor_damage/
    └── severe_damage/
```

## Dataset Requirements

- **Minimum**: Start with 10-20 images per class for MVP/prototype
- **Recommended**: 50-100+ images per class for better model performance
- **Image format**: JPG, PNG, or other PIL-supported formats
- **Image content**:
  - `healthy/`: Images of intact blades with no visible damage
  - `minor_damage/`: Images showing minor cracks, chips, or erosion
  - `severe_damage/`: Images showing significant cracks, large chips, or severe erosion

## Notes

- You can mix TPI and LM blade images together initially
- The shutdown decision logic already handles blade_type separately
- For production, consider separate datasets for TPI vs LM if they have distinct visual characteristics
- You can later add more granular labels like "leading_edge_erosion", "tip_damage", etc.

## Creating the Structure

```bash
mkdir -p data/{train,val,test}/{healthy,minor_damage,severe_damage}
```

Then place your labeled images in the appropriate directories.

