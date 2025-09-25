import matplotlib.pyplot as plt
import torch
import numpy as np

@torch.no_grad()
def plot_and_save(points, pngname, title='', axlim=None):
    points = points.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:,0], points[:,1], points[:,2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])


    # Manually enforce the same range in all directions:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    ranges = np.array([x_limits, y_limits, z_limits])
    span = ranges[:,1] - ranges[:,0]
    max_span = max(span)
    centers = np.mean(ranges, axis=1)

    # Update limits so that each axis has the same length:
    # ax.set_xlim3d(centers[0] - max_span/2, centers[0] + max_span/2)
    # ax.set_ylim3d(centers[1] - max_span/2, centers[1] + max_span/2)
    # ax.set_zlim3d(centers[2] - max_span/2, centers[2] + max_span/2)

    plt.savefig(pngname)
    print('Saving to', pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()