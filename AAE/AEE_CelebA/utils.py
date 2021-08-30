"""Contains all utils function."""
import matplotlib.pyplot as plt


def show_grid_tensor(grid_im, label, feature_idx, n, features, c=None):
    """Show the batch of Images."""
    fig = plt.figure(figsize=(15, 15))
    for i in range(n):
        ax = plt.subplot(8, 8, i+1)
        plt.imshow(grid_im[i], cmap=c)
        plt.axis("off")
        ax.set_title(f"{features[feature_idx+1]} {label[i][feature_idx]}")
    return plt, fig
