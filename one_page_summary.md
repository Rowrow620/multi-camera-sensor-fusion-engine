# Stereo Depth → Crop → CNN Classification — One-Page Summary

Date: 2026-05-17

Objective
- Build a compact pipeline that uses stereo disparity to extract near-field object crops, label them, and train a small CNN to classify the extracted crops.

Pipeline (high level)
- Depth estimation: compute disparity using OpenCV StereoSGBM/StereoBM on rectified stereo pairs (C++).
- ROI extraction: threshold disparity to isolate near-depth pixels, morphology, connected components, spatial filtering (area, width/height, aspect) → save crop images and `crops_metadata.csv`.
- Labeling: create `crop_labels.csv` (template) and materialize `crops/labeled/<class>/` via helper script.
- Training: use an image-dataset loader and a small CNN (Conv→Pool→Conv→Pool→Dense) in TensorFlow; produce learning curves, confusion matrix, and saved model.

Files and artifacts (workspace-relative)
- Code (C++): [final project/stereo_roi_finder.cpp](stereo_roi_finder.cpp)
- Trainer (Python): [final project/cnn_train.py](cnn_train.py)
- Label helper: [final project/prepare_labeled_crops.py](prepare_labeled_crops.py)
- Crop data (example): `images/cnn/` (disparity maps, crop masks, extracted ROIs)
- Saved model: `crop_cnn.keras` (project root after training)
- Plots: `images/cnn/learning_curves.jpg`, `images/cnn/confusion_matrix.jpg`

Reproduction (quick commands)
- Compile the stereo ROI extractor (example):

```bash
# from project root
g++ -o mystereo_match_test final\ project/stereo_roi_finder.cpp `pkg-config --cflags --libs opencv4`
mkdir -p "final project/crops/raw"
./mystereo_match_test fusion/example-stereo/left_0000.png fusion/example-stereo/right_0000.png \
  --no-display --crop-dir="final project/crops/raw" --crop-min-area=400 --crop-fg-percent=0.6 --crop-max=20
```

- Create labeled dataset and train (example):

```bash
# using the project's virtualenv python
../.venv/bin/python final\ project/prepare_labeled_crops.py --mode copy --labels "final project/crops/raw/crop_labels.csv" --out "final project/crops/labeled"
../.venv/bin/python final\ project/myannbest.py
```

Dataset and results (current run)
- Labeled dataset size used for the recent training: 48 images across 3 classes (after merging `bg` → `background`).
- Example training outcomes (small dataset):
  - 4-class run: 48 images — Test loss ≈ 1.2461, Test accuracy = 0.3750
  - 3-class run (merged): 48 images — Test loss ≈ 1.0100, Test accuracy = 0.3750
- Notes: dataset is small and class-imbalanced; reported metrics are indicative only.

Design notes & implementation choices
- Depth thresholding uses a fraction of the maximum valid disparity per frame to select near objects (`--crop-fg-percent`).
- Morphological opening/closing reduces speckle noise before connected-component analysis.
- Spatial filters (area, min width/height, max aspect ratio) reduce false positives and small spurious crops.
- Trainer uses `tf.keras` with a small ConvNet and `image_dataset_from_directory` for simplicity and reproducibility.

Code references
- ROI extraction, disparity thresholding, morphological filtering, connected-component analysis, crop writing, and CSV metadata export are implemented in [stereo_roi_finder.cpp](stereo_roi_finder.cpp).
- The dataset organization and label-folder creation logic (AI-assisted helper) is in [prepare_labeled_crops.py](prepare_labeled_crops.py).
- The CNN model, training loop, early stopping, evaluation, and plot saving are implemented in [cnn_train.py](cnn_train.py).

When reviewing the pipeline, inspect these files for the exact implementation details and parameters (SGBM parameters and morphological kernel sizes in `stereo_roi_finder.cpp`, dataset split and normalization in `cnn_train.py`, and CSV parsing/label assignment in `prepare_labeled_crops.py`).

Citations
- OpenCV stereo/disparity & morphology tutorials: https://docs.opencv.org/4.x/
- TensorFlow image data + training tutorial: https://www.tensorflow.org/tutorials/load_data/images

Next steps to finish the assignment
- Run a final, longer training on the full labeled set and save artifacts (`crop_cnn.keras`, `images/cnn/*`).
- Produce the single-page PDF version of this summary (optional) and a short README with exact commands and environment notes.
- Package code + artifacts into a submission archive (zip).

Contact / notes
- I generated this summary from the repository state and most recent training logs. If you want this converted to PDF or shortened further, tell me which format to produce and I'll create it.
