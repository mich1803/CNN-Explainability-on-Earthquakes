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
mean, std = spectra_stats("../preprocessed_dset/spectrograms", "../preprocessed_dset/metadata.csv")

def prepare_image_for_plot(image):
    # Converte il tensore PyTorch in un array NumPy
    
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

def compute_shap_tensor(model, sample, dim, device, max_evals=1000, masker_settings="inpaint_telea", save_dir = False):

    images, _, trace_name, *_ = sample

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
    
    shap_tensor = np.array(shap_values.values).squeeze(axis=-1)

    if save_dir:
        np.save(os.path.join(save_dir, f"{trace_name[0]}.npy"), shap_tensor) 
        
    return shap_tensor
    

def plot_mean_shap_p(mean_shap_tensor,  
              ft, 
              hist=None, 
              hist_bins = 50,
              alpha_min=None, 
              alpha_max=None, 
              title = False, 
              figsize=(15, 7),
              save_path = False):
    
    f, t = ft

    if not alpha_max:
        alpha_max = np.max(np.abs(mean_shap_tensor))
    if not alpha_min:
        alpha_min = np.min(np.abs(mean_shap_tensor))
    alpha_normalizer = max(np.abs(alpha_min), np.abs(alpha_max))

    plt.figure(figsize=figsize)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.axvline(5, c='black', label = f'p-wave arrival', lw = 3, alpha = .3)
    im = plt.imshow(
                    mean_shap_tensor, 
                    cmap="coolwarm", 
                    alpha=np.clip(np.abs(mean_shap_tensor)/alpha_normalizer, 0, 1), 
                    aspect='auto', 
                    origin='lower', 
                    extent=[*t, *f],
                    vmin=alpha_min,  # Fix color range
                    vmax=alpha_max
                    )
    if isinstance(hist, np.ndarray) and hist.size > 0:
        plt.hist(hist, bins=hist_bins, alpha=0.3, color = "black", label=f"Distribution of s-wave arrival")
    cbar = plt.colorbar(im, orientation="horizontal", pad=0.1)
    cbar.set_label("Contribution to the prediction")
    plt.legend()
    plt.title(title)

    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save if a path is provided
        plt.close()


def plot_mean_shap_s(mean_shap_tensor,  
              ft, 
              hist=None, 
              hist_bins = 50,
              alpha_min=None, 
              alpha_max=None, 
              title = False, 
              figsize=(15, 7),
              save_path = False):
    
    f, t = ft
    t = (0, 8)

    if not alpha_max:
        alpha_max = np.max(np.abs(mean_shap_tensor))
    if not alpha_min:
        alpha_min = np.min(np.abs(mean_shap_tensor))
    alpha_normalizer = max(np.abs(alpha_min), np.abs(alpha_max))

    plt.figure(figsize=figsize)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.axvline(4, c='black', label = f's-wave arrival', lw = 3, alpha = .3)
    im = plt.imshow(
                    mean_shap_tensor, 
                    cmap="coolwarm", 
                    alpha=np.clip(np.abs(mean_shap_tensor)/alpha_normalizer, 0, 1), 
                    aspect='auto', 
                    origin='lower', 
                    extent=[*t, *f],
                    vmin=alpha_min,  # Fix color range
                    vmax=alpha_max
                    )
    if isinstance(hist, np.ndarray) and hist.size > 0:
        plt.hist(hist, bins=hist_bins, alpha=0.3, color = "black", label=f"Distribution of p-wave arrival")
    cbar = plt.colorbar(im, orientation="horizontal", pad=0.1)
    cbar.set_label("Contribution to the prediction")
    plt.legend()
    plt.title(title)

    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save if a path is provided
        plt.close()



def plot_wf_shap(shap_tensor, 
                 spectrogram, 
                 waveform, 
                 alt_wave,
                 ft = None,
                 alpha_min = None,
                 alpha_max = None,
                 label = None,
                 day = None,
                 week = None,
                 name = None,
                 type = "p",
                 show = True,
                 save_path = False,
                 figsize = (15, 20)):
    spectrogram = prepare_image_for_plot(spectrogram)
    f, t = (ft[0], ft[1])

    if not alpha_max:
        alpha_max = np.max(shap_tensor)
    if not alpha_min:
        alpha_min = np.min(shap_tensor)
    alpha_normalizer = max(np.abs(alpha_min), np.abs(alpha_max))
    
    grayscale_shap_tensor = np.mean(shap_tensor, axis=-1)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1.5, 1.5, 4], 'hspace': 0.2}, sharex=True)
    fig.suptitle(f"SHAP on waveform and spectrogram\n label: {'Aftershock' if label == 'post' else 'Foreshock'},   trace_name = {name}, \n date = {day} (week: {week})")

    label_dict = {0: "HHE", 1: "HHN", 2: "HHZ"}

    maxy = np.max(waveform)

    for ch in range(3):
        axes[ch].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        axes[ch].axvline(5, c='green', lw=2, alpha=1)
        axes[ch].axvline(alt_wave, c='green', ls = "dotted", lw=2, alpha=1)

        axes[ch].plot(np.linspace(t[0], t[1], waveform[ch].shape[0]), waveform[ch], lw=1, color="black", label = label_dict[ch])  
        axes[ch].set_xlim(t)  # Ensures x-axis is properly aligned with the last plot
        axes[ch].legend(loc="upper right")
        axes[ch].set_ylim([-maxy, maxy])

    axes[1].set_ylabel("npts")
    # Plot the spectrogram in the 4th row
    
    axes[3].imshow(
        spectrogram,
        aspect='auto',
        origin='lower',
        extent=[t[0], t[1], f[0], f[1]]
    )

    im = axes[3].imshow(
        grayscale_shap_tensor, 
        cmap="coolwarm", 
        alpha=np.clip(np.abs(grayscale_shap_tensor)/alpha_normalizer, 0, 1), 
        aspect='auto', 
        origin='lower', 
        extent=[*t, *f],
        vmin=alpha_min,  # Fix color range
        vmax=alpha_max
        )
    
    axes[3].axvline(5, c='green', label=f'{type}-wave arrival', lw = 2, alpha = 1)
    axes[3].axvline(alt_wave, c='green', ls = "dotted", label=f"{'s' if type == 'p' else 'p'}-wave arrival", lw = 2, alpha = 1)
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("Frequency [Hz]")
    plt.xlabel('Time [s]')
    
    plt.colorbar(im, ax = axes, orientation='vertical', label="Contribution to the prediction")
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save if a path is provided
        plt.close(fig)  # Close the figure to free memory