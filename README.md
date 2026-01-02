![background](media/header.png)

## Overview  
This project explores the application of explainability techniques in deep learning models trained on seismic data. Inspired by the work of Laurenti et al. [[1]](https://www.nature.com/articles/s41467-024-54153-w), we focus on understanding what machine learning models learn when classifying foreshocks and aftershocks or predicting earthquake magnitude. Using SHAP (SHapley Additive exPlanations), we analyze the importance of input features derived from seismic waveforms, converted into spectrograms.

## How to Use  

1. **Download the Dataset**  
   The dataset can be downloaded from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15683047.svg)](https://doi.org/10.5281/zenodo.15683047) 

2. **Clone the repository**  
   ```bash  
   git clone https://github.com/mich1803/CNN-Explainability-on-Earthquakes/
   cd your_path/CNN-explainability-Earthquakes  
   ```

3. **Create and activate Virtual Environment**
    ```bash  
    python3 -m venv CNN_EQML
    source CNN_EQML/bin/activate
   ```

4. **Install dependecies**
    ```bash 
    pip install -r requirements.txt  
    ```

5. **Follow the Pipeline Instructions**  
   See the next section for a step-by-step guide on reproducing the pipeline.


## Pipeline Instructions

1. **Preprocessing**  
   - Run `preprocessing/preprocess.ipynb` to generate RGB spectrograms from raw waveform data in `dset/`.  
   - ⚠️ *You can skip this step by unzipping `dset_preprocessed.zip`, also available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15683047.svg)](https://doi.org/10.5281/zenodo.15683047)*

2. **Training**  
   - Use `training/training.ipynb` to train the CNN with cross-validation on the preprocessed dataset.  
   - This notebook includes a snippet to select **300 random correctly predicted samples per class** for the SHAP step.  
   - ⚠️ *You can skip this step by unzipping `training/model_checkpoints.zip` and `explainability/samples.zip`*

3. **SHAP Tensor Generation**  
   - Run `explainability/compute_shap.ipynb` to compute SHAP tensors for selected samples.  
   - ⚠️ *You can skip this by unzipping `explainability/shap/tensors.zip`*  
   - This notebook also includes code to generate the SHAP visualization plots.

4. **Dataset & SHAP Overview Plots**  
   - Use `explainability/general_plots.ipynb` to explore general statistics about the dataset and the generated SHAP tensors.


### Plots of SHAP tensor examples
  - Single SHAP with RGB event on background:
            ![single_example](media/single_example.png)
  - Mean SHAP with white background:
            ![mean_example](media/mean_example.png)

---

## Conference Paper (ICIAP 2025):
https://doi.org/10.1007/978-3-032-10185-3_25

## Zenodo Repositories
  - Code Repo:    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15747240.svg)](https://doi.org/10.5281/zenodo.15747240) 
  - Data Repo:    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15683047.svg)](https://doi.org/10.5281/zenodo.15683047) 
