import sys
import os
import torch
import pytorch_lightning as pl
import numpy as np
import random
import shap
from tqdm import tqdm
import matplotlib.pyplot as plt


# Add the 'training' directory to sys.path
sys.path.append(os.path.abspath("../training"))
from training_utils import spectra_stats

# Set a fixed seed value
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pl.seed_everything(SEED)

# Setting for the datasets
mean_64, std_64 = spectra_stats(os.path.join("../preprocessed_dset/sp_64", "train"))
mean_32, std_32 = spectra_stats(os.path.join("../preprocessed_dset/sp_32", "train"))

def prepare_image_for_plot(image, spec_type):
    # Converte il tensore PyTorch in un array NumPy
    if spec_type[-2:] == "64":
        mean = mean_64
        std = std_64
    elif spec_type[-2:] == "32":
        mean = mean_32
        std = std_32
    else:
        raise ValueError("Invalid spec_type")
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        
    return (np.clip(image.transpose(1,2,0) * std + mean, 0, 1) * 255).astype(np.uint8)

# Function to preprocess input for the model
def preprocess_for_shap(images):
    # Normalize to [0, 1] using torch functions
    min_val = torch.min(images)
    max_val = torch.max(images)
    shap_images = (images - min_val) / (max_val - min_val)
    
    # Scale to [0, 255] and convert to uint8
    shap_images = (shap_images * 255).to(torch.uint8)
    
    # Convert from (batch_size, 3, 33, 153) to (batch_size, 33, 153, 3)
    shap_images = shap_images.permute(0, 2, 3, 1)
    
    return shap_images, min_val.item(), max_val.item()

def inverse_preprocess_for_shap(shap_images, original_min, original_max):
    # Convert from (batch_size, 33, 153, 3) to (batch_size, 3, 33, 153)
    shap_images = shap_images.permute(0, 3, 1, 2)
    
    # Scale from [0, 255] to [0, 1]
    shap_images = shap_images / 255.0
    
    # Rescale to the original range [original_min, original_max]
    shap_images = shap_images * (original_max - original_min) + original_min
    
    return shap_images

def compute_shap_tensor(model, sample, dim, device, max_evals=1000, masker_settings="inpaint_telea"):

    images, _, _ = sample

    def model_fn(images, device = device):
        # Convert from NumPy array to PyTorch tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        # Ensure the images are moved to the correct device
        images = images.to(device)
        
        # Perform inverse preprocessing for SHAP
        images = inverse_preprocess_for_shap(images, original_min, original_max)
        
        # Pass the images through the model
        outputs = model(images)
        
        # Return the outputs as NumPy array
        return outputs.cpu().detach().numpy()

    images_to_explain, original_min, original_max = preprocess_for_shap(images)
    masker = shap.maskers.Image(masker_settings, (*dim, 3))
    explainer = shap.Explainer(model_fn, masker, output_names=["Foreshock", "Aftershock"])

    shap_values = explainer(
        images_to_explain,
        max_evals=max_evals,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:1]
        )
    
    return np.array(shap_values.values).squeeze(axis=-1)

def compute_mean_shap_tensor(model, dloader, dim, device, max_evals = 1000, masker_settings = "inpaint_telea", save_path = None):

    tot = len(dloader)
    mean_shap_tensor = np.zeros((*dim, 3))

    for sample in tqdm(dloader, desc = "Computing Mean SHAP Tensor", total = tot):
        mean_shap_tensor += np.mean(compute_shap_tensor(model, sample, dim, device, max_evals, masker_settings), axis=0)/tot

    if save_path:
        np.save(save_path, mean_shap_tensor)

    return mean_shap_tensor
    

def plot_shap(shap_tensor, onechannel, background, ft, alpha_normalizer=None, label=None, name=None, model_output=None, spec_type="p64", mean = False, figsize=(15, 7)):
    f, t = (ft[0], ft[1]) if spec_type[-2:] else (ft[2], ft[3]) # ft = [f64, t64, f32, t32]
    if spec_type[0] == 's':
        t = (0,20)
    if onechannel:
        grayscale_shap_tensor = np.mean(shap_tensor, axis=-1)
        if not alpha_normalizer:
            alpha_normalizer = np.max(np.abs(grayscale_shap_tensor))
        plt.figure(figsize=figsize)
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.axvline(5, c='black', label = f'{spec_type[0]}-wave arrival')
        if background != None:
            bg_image = prepare_image_for_plot(background, spec_type)
            plt.imshow(
                        bg_image, 
                        aspect='auto', 
                        origin='lower', 
                        extent=[*t, *f]
                        )
        im = plt.imshow(
                        grayscale_shap_tensor, 
                        cmap="coolwarm", 
                        alpha=np.abs(grayscale_shap_tensor)/alpha_normalizer, 
                        aspect='auto', 
                        origin='lower', 
                        extent=[*t, *f]
                        )
        cbar = plt.colorbar(im, orientation="horizontal", pad=0.1)
        cbar.set_label("Contribution to the prediction")
        plt.legend()
        if not mean:
            plt.title(f"SHAP on 1-channel mean ({name})\n Model output: {model_output.cpu().detach().numpy()} --> Label: {'Aftershock' if label else 'Foreshock'}")
        else:
            plt.title(mean)
        plt.show()

    if not onechannel:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        if not alpha_normalizer:
            alpha_normalizer = np.max(np.abs(grayscale_shap_tensor))
        if not mean:
            plt.suptitle(f"SHAP on the three components ({name})\n Model output: {model_output.cpu().detach().numpy()} --> Label: {'Aftershock' if label else 'Foreshock'}")
        else:
            plt.suptitle(mean)
        for i in range(3):
            if background != None:
                bg_image = prepare_image_for_plot(background, spec_type)
                axes[i].imshow(
                                bg_image[:,:,i],  # Ensure correct shape
                                cmap="gray",
                                aspect="auto",
                                origin="lower",
                                extent=[*t, *f],
                            )
            im = axes[i].imshow(
                                shap_tensor[:,:,i],
                                cmap = "coolwarm",
                                alpha = np.abs(shap_tensor[:,:,i])/alpha_normalizer,
                                aspect = "auto",
                                origin = "lower",
                                extent = [*t, *f]
                                )
            axes[i].set_ylabel("Freuqency (Hz)")
            axes[i].axvline(5, c='black', label=f'{spec_type[0]}-wave arrival')
            axes[i].set_yticks(np.linspace(*f, num=6))  # Adjust number of ticks as needed
            axes[i].set_yticklabels([f"{freq:.1f}" for freq in np.linspace(*f, num=6)])  # Format frequency labels

        plt.colorbar(im, ax = axes, orientation='vertical', label="Contribution to the prediction")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("Time (s)")

