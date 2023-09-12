"""
This is the baseline obtained from "Input complexity and out-of-distribution detection with 
likelihood-based generative models" by Serra et al. https://arxiv.org/abs/1909.11480

They perform a straightforward method for OOD detection where they consider the compression
size needed as a proxy for the negative entropy term to correct the likelihood.

They name this value L and the score "S" is then equal to "- likelihood - S".

"""


from ..base_method import OODBaseMethod
import torch
import typing as th
import numpy as np
from ..visualization import visualize_histogram
from tqdm import tqdm
import cv2
import math
from math import inf
import os
from PIL import Image 
import subprocess

def _get_filename():
    # This piece of code makes concurrency possible for a machine
    return f"COMPLEXITY_BASELINE_{os.getpid()}"

class CompelxityBased(OODBaseMethod):
    """
    This OOD detection method visualizes trends of the latent statistics that are being calculated in the ood.methods.linear_approximations.latent_statistics.
    
    You specify a latent_statistics_calculator_class and a latetn_statistics_calculator_args and it automatically instantiates a latent statistics calculator.
    
    """
    def __init__(
        self,
        likelihood_model: torch.nn.Module,
        
        x_loader: th.Optional[torch.utils.data.DataLoader] = None,
        in_distr_loader: th.Optional[torch.utils.data.DataLoader] = None,
        
        #
        compression_type: th.Literal["ensemble", "PNG", "JPEG2000", "FLIF"] = "ensemble",
        
        
        # for logging args
        verbose: int = 0,
        
        # 
        additional_compression_args: th.Optional[th.Dict[str, th.Any]] = None,
        
        # the coefficient for the correction
        correction_coeff: float = 1.0,
    ):
        super().__init__(
            likelihood_model = likelihood_model,
            x_loader=x_loader,
            in_distr_loader=in_distr_loader,
        )
        self.verbose = verbose

        # disable all the parameters in the model
        self.likelihood_model.eval()
        
        # get rid of all the randomness in the log_prob
        self.likelihood_model.denoising_sigma = False
        self.likelihood_model.dequantize = False
        
        # iterate over all the parameters of likelihood_model and turn off their gradients
        # for faster performance
        for param in self.likelihood_model.parameters():
            param.requires_grad = False
        
        self.compression_type = compression_type
        self.verbose = verbose
        self.correction_coeff = correction_coeff
        self.additional_compression_args = additional_compression_args or {}
    
    def png_compressor(self, x: np.array, compression_level: int = 9):
        """
        Perform a compression using the cv2 compressor
        As per the details of the method, we set the compression level
        to the maximum value possible.
        """
        img_encoded = cv2.imencode('.png',x,[int(cv2.IMWRITE_PNG_COMPRESSION),compression_level])
        
        return 8.0 * len(img_encoded[1])
    
    def jpeg2000_compressor(self, x: np.array, compression_level: int = 9):
        """
        Perform a compression using the cv2 compressor
        As per the details of the method, we set the compression level
        to the maximum value possible.
        """
        cv2.imwrite(f"{_get_filename()}_tmp.jpg", x, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_level])
        ret = 8.0 * os.path.getsize(f"{_get_filename()}_tmp.jpg")
        os.remove(f"{_get_filename()}_tmp.jpg")
        return ret
    
    def flif_compressor(self, x: np.array):
        """
        Perform an FLIF compression which is typically stronger than the other 
        two types: https://flif.info/
        """
        # NOTE: make sure you have FLIF installed on your system
        # for an ubuntu machine which we ran our experiments on
        # you can install using `sudo snap install flif`
        if len(x.shape) == 2 or x.shape[-1] == 1:
            if x.shape[-1] == 1:
                x = x[:, :, 0]
            im = Image.fromarray(x.astype('uint8'), mode='L')
        elif len(x.shape) == 3:
            im = Image.fromarray(x.astype('uint8'), mode='RGB')
        im.save(f'{_get_filename()}_compression_method.png')
        subprocess.call(["flif", f"{_get_filename()}_compression_method.png", f"{_get_filename()}_compression_method.flif"])
        ret = 8.0 * os.path.getsize(f"{_get_filename()}_compression_method.flif")
        os.remove(f"{_get_filename()}_compression_method.png")
        os.remove(f"{_get_filename()}_compression_method.flif")
        return ret
    
    def run(self):
            
        
        scores = []
        
        if self.verbose > 0:
            loader_decorated = tqdm(self.x_loader, desc="computing scores for batch", total=len(self.x_loader))
        else:
            loader_decorated = self.x_loader
            
        scores = None
        complexity_scores = None
        log_likelihoods = None
        
        for x_batch in loader_decorated:
            
            with torch.no_grad():
                log_likelihoods = self.likelihood_model.log_prob(x_batch).cpu().numpy().flatten()
            
            batch_scores = []
            for chunk_idx, x in enumerate(x_batch):
                if self.verbose > 0:
                    loader_decorated.set_description(f"Compressor on chunk [{chunk_idx + 1}/{len(x_batch)}]")
                x = x.cpu().numpy()
                if len(x.shape) == 3:
                    x = np.transpose(x, (1, 2, 0))
                
                compression_score = inf
                if self.compression_type == "PNG" or self.compression_type == "ensemble":
                    compression_score = min(compression_score, self.png_compressor(x, **self.additional_compression_args))
                if self.compression_type == "JPEG2000" or self.compression_type == "ensemble":
                    compression_score = min(compression_score, self.jpeg2000_compressor(x, **self.additional_compression_args))
                if self.compression_type == "FLIF" or self.compression_type == "ensemble":
                    compression_score = min(compression_score, self.flif_compressor(x, **self.additional_compression_args))
                batch_scores.append(compression_score)

            complexity_batch_scores = np.array(batch_scores)
            likelihoods_batch = log_likelihoods / math.log(2)
            log_likelihoods = likelihoods_batch if log_likelihoods is None else np.concatenate([log_likelihoods, likelihoods_batch])
            batch_scores = likelihoods_batch + self.correction_coeff * complexity_batch_scores
            scores = batch_scores if scores is None else np.concatenate([scores, batch_scores])  
            complexity_scores = complexity_batch_scores if complexity_scores is None else np.concatenate([complexity_scores, complexity_batch_scores])
        
        visualize_histogram(
            log_likelihoods,
            plot_using_lines=True,
            bincount=25,
            x_label="Likelihood",
        )
        
        visualize_histogram(
            scores,
            plot_using_lines=True,
            bincount=25,
            x_label="Score",
        )
        
        visualize_histogram(
            complexity_scores,
            plot_using_lines=True,
            bincount=25,
            x_label="Compression",
        )
        
        