# Summary
Effectively distinguishing between images in high visual similarity datasets poses significant challenges, especially with photometric variations, perspective transformations, and/or occlusions. We introduce a novel methodology that fuses local and global feature detection techniques. By integrating local feature analysis with global feature representation based on graph structuring and processing, our approach can capture topological and metric relationships among descriptors. The proposed graph representation is computed using only matching features, hence filtering irrelevant information and focusing on unique image attributes that favor identification. This study aims to answer how the synergistic combination of these techniques can outperform conventional identification methods dealing with data sets with high visual similarity. We performed experiments showing significant improvements in precision and recall, reflected in the F1-Score, of the proposed strategy over pure local-based image identification. The results highlight the potential of hybrid approaches for better image recognition, also revealing that local-based method can use our proposal as an additional component for obtaining improved results.

# Step-by-Step Guide to Run the FusionID Project
This guide provides the necessary steps to set up and run the FusionID project on your local machine, aimed at enhancing image identification through a hybrid approach that integrates local and global feature detection.

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
sudo python3 -m venv ./venv
source ./venv/bin/activate (linux)

.\venv\Scripts\activate (windows)

pip3 install -r requirements.txt
```

## 4. Deactivate the Virtual Environment (Optional)
Once you are done, you can deactivate the virtual environment by running:
```bash
deactivate
```