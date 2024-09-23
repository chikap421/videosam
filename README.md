# VideoSAM: A Large Vision Foundation Model for High-Speed Video Segmentation

![High-Speed Video](https://img.shields.io/badge/⚡%20High--Speed%20Video-red?style=for-the-badge&logo=video&logoColor=white)
![Deep Learning](https://img.shields.io/badge/🧠%20Deep%20Learning-blue?style=for-the-badge&logo=deeplearning&logoColor=white)
![Bubble Segmentation](https://img.shields.io/badge/💧%20Bubble%20Segmentation-blue?style=for-the-badge&logo=bubble&logoColor=white)
![Multi-Modality Analysis](https://img.shields.io/badge/🔍%20Multi--Modality%20Analysis-yellow?style=for-the-badge&logo=analysis&logoColor=white)
![Dynamic Fluids](https://img.shields.io/badge/💧%20Dynamic%20Fluids-blue?style=for-the-badge&logo=water&logoColor=white)
![Patchification](https://img.shields.io/badge/🧩%20Patchification-purple?style=for-the-badge&logo=puzzle-piece&logoColor=white)
![IoU Metrics](https://img.shields.io/badge/📊%20IoU%20Metrics-green?style=for-the-badge&logo=metrics&logoColor=white)
![Composite Frames](https://img.shields.io/badge/🎥%20Composite%20Frames-orange?style=for-the-badge&logo=film&logoColor=white)
![CNN Comparison](https://img.shields.io/badge/🖥️%20CNN%20Comparison-red?style=for-the-badge&logo=data-analysis&logoColor=white)
![MIT License](https://img.shields.io/badge/📜%20License-lightgrey?style=for-the-badge&logo=open-source-initiative&logoColor=black)

## Overview

VideoSAM is a large-scale vision foundation model that excels at high-speed video segmentation, particularly in dynamic fluid environments. The model was rigorously tested across different data modalities including **Argon**, **Nitrogen**, **FC-72**, and **Water**. VideoSAM uses a **patchification** process for detailed segmentation and was capable of high-accuracy bubble segmentation across different data types.

Key features include:

- **High-speed video frame analysis** with segmentation into smaller patches for detailed analysis.
- **Mask extraction pipeline** for both single and composite frames, ensuring dynamic video data is consistently evaluated.
- **Metrics evaluation** using **IoU**, **F1 Score**, and **Precision** across frames and sequences for robust performance insights.
- **Zero-shot generalization** on unseen data modalities, showcasing VideoSAM's adaptability to various types of fluid dynamics.

---

## Key Experiments

1. **Zero-Shot Generalization Across Modalities**:
    - VideoSAM was trained on **Argon** data and tested across other modalities like **Nitrogen**, **FC-72**, and **Water**. It demonstrated superior segmentation in complex fluids, especially with intricate bubble boundaries.

2. **Performance Across Multiple Modalities**:
    - The model was trained on multiple datasets and consistently outperformed baseline models like SAM, particularly excelling in fluids with complex dynamics such as **Nitrogen** and **FC-72**.

3. **Comparison with U-Net CNN**:
    - VideoSAM was benchmarked against **U-Net**, a traditional CNN architecture. While U-Net performed better on simpler datasets like **Water**, VideoSAM surpassed it in handling more dynamic and complex fluid environments.

---

## Data Location

All training and testing datasets for VideoSAM are located in the following directories within the project structure:

- **Training Data**: `data/train/`
    - You can find the split training dataset files in this directory, named as `train_image_masks_part_aa`, `train_image_masks_part_ab`, etc.
- **Testing Data**: `data/test/`
    - Similarly, the split testing dataset files are located in this directory, with the filenames following the same convention, `test_image_masks_part_aa`, `test_image_masks_part_ab`, etc.

To reconstruct the data from its split files, follow the instructions below.

---

## How to Unpack Split Zip Files

To reassemble files that were split into smaller parts, follow these steps:

1. Navigate to the directory where the split files are located.
2. Use the `cat` command to combine them:
    ```bash
    cat train_image_masks_part_* > train_image_masks.zip
    ```
3. Unzip the combined file:
    ```bash
    unzip train_image_masks.zip
    ```

Apply the same process for test data or any other split zip files.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/chikap421/videosam.git
    cd videosam
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the environment:
    ```bash
    python setup.py install
    ```

---

## Inference Pipeline

The **inference pipeline** for VideoSAM was designed to evaluate its performance across different data modalities. This includes:

1. **Grayscale Conversion and Normalization**:
    - Frames are first converted to grayscale and normalized.

2. **Patchification**:
    - For both single and composite frames, the dataset is segmented into smaller patches using a grid-based bounding box.

3. **Mask Extraction**:
    - Patches are processed through both VideoSAM and SAM models, and the predicted masks are stitched together to reconstruct full-image masks.

4. **Metrics Evaluation**:
    - **IoU**, **F1 Score**, and **Precision** metrics are used for both single-frame and sequence-based performance evaluation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

![MIT License](https://img.shields.io/badge/📜%20License-lightgrey?style=for-the-badge&logo=open-source-initiative&logoColor=black)
