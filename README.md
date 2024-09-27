# Summary 
Effectively distinguishing between images in high visual similarity datasets poses significant challenges, especially with photometric variations, perspective transformations, and/or occlusions. We introduce a novel methodology that fuses local and global feature detection techniques. By integrating local feature analysis with global feature representation based on graph structuring and processing, our approach can capture topological and metric relationships among descriptors. The proposed graph representation is computed using only matching features, hence filtering irrelevant information and focusing on unique image attributes that favor identification. This study aims to answer how the synergistic combination of these techniques can outperform conventional identification methods dealing with data sets with high visual similarity. We performed experiments showing significant improvements in precision and recall, reflected in the F1-Score, of the proposed strategy over pure local-based image identification. The results highlight the potential of hybrid approaches for better image recognition, also revealing that local-based method can use our proposal as an additional component for obtaining improved results. 

# Step-by-Step Guide to Run the Project

Follow these steps to set up and run the project on your local machine:

## 1. Clone the Repository

First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/binarycode11/FusionID.git)
```

## 2. Navigate to the Project Directory
After cloning the repository, navigate to the project directory:
```bash
cd FusionID
```

## 3. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies. Run the following command to create one:

```bash
python3 -m venv ./venv
source ./venv/bin/activate (linux)

.\venv\Scripts\activate (windows)
```

## 4. Deactivate the Virtual Environment (Optional)
Once you are done, you can deactivate the virtual environment by running:
```bash
deactivate
```

## 5. Datasets
### Download the "Woods Texture" dataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1DzJYC00lcZo-SQWdaRQHylMGTd-Mcz2h' -O data/woods_texture.zip

### Download the "Flowers" dataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1z4Us0tlRrNEDSHlWwU42IFX57BNj7mcb' -O data/flowers.zip

### Unzip the downloaded files
unzip data/woods_texture.zip -d data/woods_texture
unzip data/flowers.zip -d data/flowers

### Remove the zip files after extraction (optional)
rm data/woods_texture.zip
rm data/flowers.zip


# Results for Flowers Dataset

| Num Features | Feature Local Class  | Distance | Threshold | Matches                  | Scores                   |
|--------------|----------------------|----------|-----------|--------------------------|--------------------------|
| 30           | GFTTAffNetHardNet     | 0.9      | 0.5       | [TP:444,FP:56,FN:66]     | [TP:486,FP:46,FN:24]     |
|              |                      |          |           | [P:0.89,R:0.87,F1:0.88]  | [P:0.91,R:0.95,F1:0.93]  |
| 30           | GFTTFeatureSosNet     | 0.9      | 0.5       | [TP:451,FP:46,FN:59]     | [TP:489,FP:27,FN:21]     |
|              |                      |          |           | [P:0.91,R:0.88,F1:0.90]  | [P:0.95,R:0.96,F1:0.95]  |
| 30           | KeyNetHardNet         | 0.9      | 0.5       | [TP:348,FP:31,FN:162]    | [TP:408,FP:4,FN:102]     |
|              |                      |          |           | [P:0.92,R:0.68,F1:0.78]  | [P:0.99,R:0.80,F1:0.89]  |
| 30           | HesAffNetHardNet      | 0.9      | 0.5       | [TP:409,FP:63,FN:101]    | [TP:469,FP:59,FN:41]     |
|              |                      |          |           | [P:0.87,R:0.80,F1:0.83]  | [P:0.89,R:0.92,F1:0.90]  |
| 30           | SIFTFeature           | 0.9      | 0.5       | [TP:450,FP:312,FN:60]    | [TP:496,FP:191,FN:14]    |
|              |                      |          |           | [P:0.59,R:0.88,F1:0.71]  | [P:0.72,R:0.97,F1:0.83]  |
| 30           | SIFTFeatureScaleSpace | 0.9      | 0.5       | [TP:432,FP:506,FN:78]    | [TP:489,FP:503,FN:21]    |
|              |                      |          |           | [P:0.46,R:0.85,F1:0.60]  | [P:0.49,R:0.96,F1:0.65]  |
| 30           | SIFTFeature           | 0.8      | 0.5       | [TP:425,FP:12,FN:85]     | [TP:474,FP:2,FN:36]      |
|              |                      |          |           | [P:0.97,R:0.83,F1:0.90]  | [P:1.00,R:0.93,F1:0.96]  |
| 30           | SIFTFeatureScaleSpace | 0.8      | 0.5       | [TP:445,FP:38,FN:65]     | [TP:488,FP:10,FN:22]     |
|              |                      |          |           | [P:0.92,R:0.87,F1:0.90]  | [P:0.98,R:0.96,F1:0.97]  |


# Results for Wood Textures Dataset

| Num Features | Feature Local Class  | Distance | Threshold | Matches                  | Scores                   |
|--------------|----------------------|----------|-----------|--------------------------|--------------------------|
| 30           | GFTTAffNetHardNet     | 0.9      | 0.5       | [TP:23,FP:3,FN:154]      | [TP:39,FP:0,FN:138]      |
|              |                      |          |           | [P:0.88,R:0.13,F1:0.23]  | [P:1.00,R:0.22,F1:0.36]  |
| 30           | GFTTFeatureSosNet     | 0.9      | 0.5       | [TP:29,FP:0,FN:148]      | [TP:46,FP:1,FN:131]      |
|              |                      |          |           | [P:1.00,R:0.16,F1:0.28]  | [P:0.98,R:0.26,F1:0.41]  |
| 30           | KeyNetHardNet         | 0.9      | 0.5       | [TP:0,FP:3,FN:177]       | [TP:5,FP:0,FN:172]       |
|              |                      |          |           | [P:0.00,R:0.00,F1:0.00]  | [P:1.00,R:0.03,F1:0.05]  |
| 30           | HesAffNetHardNet      | 0.9      | 0.5       | [TP:29,FP:4,FN:148]      | [TP:43,FP:1,FN:134]      |
|              |                      |          |           | [P:0.88,R:0.16,F1:0.28]  | [P:0.98,R:0.24,F1:0.39]  |
| 30           | SIFTFeature           | 0.9      | 0.5       | [TP:15,FP:57,FN:162]     | [TP:44,FP:0,FN:133]      |
|              |                      |          |           | [P:0.21,R:0.08,F1:0.12]  | [P:1.00,R:0.25,F1:0.40]  |
| 30           | SIFTFeatureScaleSpace | 0.9      | 0.5       | [TP:4,FP:131,FN:173]     | [TP:9,FP:0,FN:168]       |
|              |                      |          |           | [P:0.03,R:0.02,F1:0.03]  | [P:1.00,R:0.05,F1:0.10]  |
| 30           | SIFTFeature           | 0.8      | 0.5       | [TP:6,FP:0,FN:171]       | [TP:34,FP:0,FN:143]      |
|              |                      |          |           | [P:1.00,R:0.03,F1:0.07]  | [P:1.00,R:0.19,F1:0.32]  |
| 30           | SIFTFeatureScaleSpace | 0.8      | 0.5       | [TP:1,FP:14,FN:176]      | [TP:8,FP:0,FN:169]       |
|              |                      |          |           | [P:0.07,R:0.01,F1:0.01]  | [P:1.00,R:0.05,F1:0.09]  |
