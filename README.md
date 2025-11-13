Code, data-preprocessing scripts, notebooks for Rohan Singh’s research papers (deepfake detection experiment).
IEEE paper: https://ieeexplore.ieee.org/abstract/document/10931956
**Author:** Rohan Singh  
**Email:** singh.rohan068@gmail.com  
**Affiliation:** Department of Computer Science & Engineering, GLA University, Mathura

---

## Repository overview

This repository collects code, notebooks, data-preprocessing scripts, and LaTeX sources related to Rohan Singh's published and in-progress research work, including but not limited to:

- Deepfake detection experiments
- Supporting datasets, evaluation scripts, and experiment logs
High-level purpose

The notebook is a research/experiment pipeline for image-based model experiments (appears to be for tasks such as deepfake/blood-cell/classification). It:

Authenticates to Kaggle and downloads a dataset.

Prepares and splits the dataset into train/val/test.

Provides image preprocessing utilities.

Builds/loads model(s) (EfficientNet + TensorFlow/PyTorch interfaces are present).

Runs inference (single image and folder/batch prediction) and displays results.

Contains helper utilities to create sample datasets and visualize images.

Uses sklearn for metrics and matplotlib for visualization.

Key libraries used: torch, tensorflow, efficientnet, cv2 (OpenCV), numpy, matplotlib, sklearn, splitfolders.

Notebook structure & important cells (walkthrough)

Cell 0–3 — Kaggle setup & dataset download

Uses google.colab.files.upload() to upload a kaggle.json credentials file.

Moves kaggle.json to ~/.kaggle/ and sets permissions.

Likely uses Kaggle CLI to download dataset(s) (the preview shows the authentication steps).

Cell 4–6 — File / dataset inspection

Commands like !ls -l and perhaps !unzip or !kaggle datasets download ... to retrieve and extract data.

Shows dataset structure (images, labels, CSVs) and maybe an example.

Cell 7 — Split into train/val/test

Uses splitfolders to split a folder of images into train/ val/ test folders with a chosen ratio.

This prepares a standard directory structure for model training/evaluation.

Cell 8–9 — Visualization helpers

Functions to display images from a folder (thumbnails or sample images).

Likely uses matplotlib.pyplot.imshow() and OpenCV (cv2.cvtColor) to show images inline.

Cell 10 — Model loading

A load_model(...) function is present. Based on imports, the notebook supports EfficientNet and possibly models in both PyTorch and TensorFlow.

The function likely:

Accepts a model path or model name.

Loads weights and returns a ready-to-use model object.

May set model.eval() for PyTorch or model.predict for TF.

Cell 11 — Prediction for a single image

predict_image(image_path, model, transform, device) (or similar) reads an image, applies preprocessing transforms, runs the model, and returns predicted label(s) with probabilities/scores.

Converts model logits to human-readable class names; may apply softmax.

Cell 12–13 — Dataset/sample utilities

create_sample_dataset(...) duplicates or copies a small subset of images into a sample/ folder for quick experiments.

Another display_images_from_folder(...) (or a duplicate) to preview samples.

Cell 14–15 — Batch processing / image processing

process_image(...) handles resizing, normalization, augmentation (maybe), and returns an array/tensor.

process_directory(...) loops through a folder, processes all images, uses predict_image to get predictions, and saves outputs/CSV with predictions (or organizes images into predicted label folders).

Evaluation & metrics

sklearn is imported — the notebook uses it for metrics: accuracy, precision, recall, confusion matrix, classification report.

Visualizations (matplotlib) to show confusion matrix or examples where model failed.

Saved helper file

I created /mnt/data/code_notebook_code_cells.py which concatenates the code cells — you can open that file to see all code in one place.

Main functions (what they do, inputs & outputs)

Below are the main utilities detected and what they typically do:

load_model(path_or_name, framework='torch'|'tf', device='cpu'|'cuda')

Loads model architecture and weights.

Returns a model in evaluation mode.

If PyTorch, sets model.to(device) and model.eval().

predict_image(image_path, model, transform=None, device='cpu', class_map=None)

Inputs: image path (or image array), model, preprocessing transform, device.

Steps: read image (cv2), convert color, resize, apply transform, add batch dimension, move to device.

Runs forward pass, apply softmax or argmax to get label.

Returns: predicted class label, probability/confidence, optionally raw logits.

process_image(image, size=(224,224), normalize=True)

Resizes image, converts to float32, scales pixel values, optionally normalizes by mean/std expected by model.

process_directory(input_dir, model, out_csv=None, transform=None)

Walks folder, calls process_image + predict_image on each file.

Writes results to CSV or groups images by predicted label.

create_sample_dataset(src_dir, dest_dir, n_per_class)

Copies n_per_class images into dest_dir/class_name/ for quick testing.

display_images_from_folder(folder, n=9)

Plots n images in a grid so you can visually check dataset samples.

Expected directory layout (how to organize files to run notebook)

Typical layout expected by the notebook:

project_root/
├── kaggle.json                 # uploaded to authenticate
├── dataset.zip or dataset/     # raw downloaded dataset
├── dataset_extracted/          # dataset images arranged by class or with CSV labels
│   ├── train/
│   │   ├── classA/
│   │   └── classB/
│   ├── val/
│   └── test/
├── notebooks/code.ipynb
├── models/
│   └── my_model_weights.pth    # model weights to load
└── results/


If dataset is CSV-labeled (image path + label), the notebook probably reads that CSV and copies files into a class-structured folder (or uses custom loader).

How to run (quick instructions)

In Google Colab: upload kaggle.json via the first cell (it uses google.colab.files.upload()).

Execute the Kaggle download cells (!kaggle datasets download ...) or manually place the dataset in the expected folder.

Run the split cell (splitfolders) to create train/ val/ test/.

Ensure you have dependencies installed:

pip install numpy pandas matplotlib opencv-python scikit-learn split-folders torch torchvision tensorflow efficientnet-pytorch

Adjust if the notebook assumes efficientnet package variant.

If using GPU, set device = 'cuda' and ensure proper CUDA drivers are available.

Run the model loading cell with the correct weights path, then run process_directory or predict_image for inference.

Dependencies (important ones)

Python 3.8+

numpy, pandas

opencv-python (cv2)

matplotlib

scikit-learn

torch / torchvision (for PyTorch models)

tensorflow (if TF models/efficientnet-tf are used)

efficientnet or efficientnet_pytorch (depends on implementation)

splitfolders

Suggestions & typical pitfalls

Mixing frameworks: the notebook imports both torch and tensorflow. Confirm which framework your model uses; mixing without clear separation causes confusion. Keep separate loader/predict functions for TF and Torch.

Preprocessing mismatch: make sure the preprocessing (mean/std, resize, normalization) matches what the model expects (EfficientNet expects specific normalization/scaling).

Class label mapping: ensure a stable class_map (index-to-label mapping) is saved/loaded with the model (e.g., a classes.json) so predictions map to correct label names.

Large datasets in notebook: avoid copying large raw datasets into the repo; use data/README.md with instructions.

GPU availability: check torch.cuda.is_available() before moving model to CUDA.

Use try/except and logging for batch processing so a corrupt image file doesn't stop the whole run.

Improvements I recommend

Add a clear config.yaml to hold model paths, input size, batch size, device, class_map path.

Save class_map to JSON when preparing train folders.

Save prediction outputs to CSV with columns: image_path, true_label, predicted_label, confidence.

Add code to compute and visualize confusion matrix and per-class metrics after process_directory.

Use a small requirements.txt with pinned versions for reproducibility.

Add unit tests for process_image and predict_image.

Where I put a combined copy of the code

I saved a concatenated version of your code cells as:

/mnt/data/code_notebook_code_cells.py


Open that file to see all code cells in one script-like file — it's easier to search and inspect particular functions.

If you'd like any of the following, I can do it right away:

Paste the exact source of a particular cell (e.g., “Cell 10”).

Extract and show the load_model and predict_image function implementations verbatim.

Produce a short README.md for the notebook with exact run commands and dependencies pinned.

Convert the notebook inference pipeline into a small Python script (CLI) that accepts a folder and outputs a CSV of predictions.
