"""
This code performs PCA transformation on flattened data to visaulize them.
"""
from torch.utils.data import DataLoader
import typing as th
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
from PIL import Image

def print_data_2d(
    list_of_loaders: th.List[DataLoader],
    list_of_labels: th.List[str],
    alphas: th.Optional[th.List[float]] = None,
    colors: th.Optional[th.List] = None,
    pca_idx: int = 0,
    batch_limit: th.Optional[int] = None,
    close_matplotlib: bool = False,
):
    if alphas is None:
        alphas = [1.0 for _ in range(len(list_of_loaders))]
        
    list_of_data = []
    
    for loader in list_of_loaders:
        X = []
        for x_ in loader:
            if isinstance(x_, tuple) or isinstance(x_, list):
                x = x_[0]
            else:
                x = x_
            
            x = x.detach().cpu().numpy()
            X.append(x.reshape(x.shape[0], -1))
            if batch_limit is not None and len(X) > batch_limit:
                break
        list_of_data.append(np.concatenate(X))
        
    
    
    if X[0].shape[1] != 2:
        pca = PCA(n_components=2)
        pca.fit(list_of_data[pca_idx])
    
    if colors is None:
        colors = [None for _ in list_of_labels]
        
    for label, alpha, X, col in zip(list_of_labels, alphas, list_of_data, colors):
        if X.shape[1] != 2:
            X2d = pca.transform(X)
        else:
            X2d = X
        plt.scatter(X2d[:, 0].reshape(-1), X2d[:, 1].reshape(-1), c=col, label=label, alpha=alpha)
    plt.legend()
    plt.title("2D visualization of data") 
    # Save the plot to a buffer (in memory)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Open the image and convert to a numpy array
    img = Image.open(buf)
    img_array = np.array(img)

    # Close buffer
    buf.close()
    if close_matplotlib:
        plt.close()
    # Return the numpy array
    return img_array


def print_heatmap(
    loader_data,
    loader_estimate,
    batch_limit: th.Optional[int] = None,
    close_matplotlib: bool = False,
    averaging_r: th.Optional[float] = None,
):
    
    X = []
    est = []
    idx = 0
    for x_, estimate in zip(loader_data, loader_estimate):
        idx += 1
        if idx > batch_limit:
            break
        if isinstance(x_, tuple) or isinstance(x_, list):
            x = x_[0]
        else:
            x = x_
        X.append(x.reshape(x.shape[0], -1).cpu().numpy())
        est.append(estimate)
        
    X = np.concatenate(X)
    est = np.concatenate(est)
    
    
    if averaging_r is not None:
        z = np.zeros(len(est))
        # Compute the average 'y' within the radius 'r' for each point
        for i in range(len(X)):
            # Compute the distance from the current point to all other points
            distances = np.sqrt(((X - X[i, :])**2).sum(axis=1))
            
            # Identify points within the radius 'r'
            within_radius = distances < averaging_r
            
            # Compute the average 'y' of these points
            z[i] = est[within_radius].mean()
        est = z
    
    if X.shape[1] != 2:
        pca = PCA(n_components=2)
        pca.fit(X)
        X = pca.transform(X)
    
    scatter = plt.scatter(X[:, 0].reshape(-1), X[:, 1].reshape(-1), marker='^', c=est, cmap='viridis', alpha=0.5)
    # Adding a colorbar to show the intensity scale
    plt.colorbar(scatter)
    plt.title("2D visualization of data estimations") 
    # Save the plot to a buffer (in memory)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Open the image and convert to a numpy array
    img = Image.open(buf)
    img_array = np.array(img)

    # Close buffer
    buf.close()
    if close_matplotlib:
        plt.close()
    # Return the numpy array
    return img_array

