import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO
from PIL import Image


def _set_equal_axes(ax):
    # Keep the 3D plot balanced while letting each frame fit itself.
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    ranges = np.array([x_limits, y_limits, z_limits])
    span = ranges[:, 1] - ranges[:, 0]
    max_span = max(span)
    centers = np.mean(ranges, axis=1)
    half_span = max_span / 2.0
    ax.set_xlim3d(centers[0] - half_span, centers[0] + half_span)
    ax.set_ylim3d(centers[1] - half_span, centers[1] + half_span)
    ax.set_zlim3d(centers[2] - half_span, centers[2] + half_span)


@torch.no_grad()
def render_frame(points, title='', axlim=None):
    points = points.detach().cpu().numpy()
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])

    _set_equal_axes(ax)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    image = Image.open(buffer).convert('RGB')
    image.load()
    buffer.close()
    limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
    plt.close(fig)
    return image, limits


def save_gif(frames, gifname, duration=250, loop=0):
    if not frames:
        raise ValueError('save_gif requires at least one frame')

    first, *rest = frames
    first.save(
        gifname,
        save_all=True,
        append_images=rest,
        duration=duration,
        loop=loop,
    )
    print('Saving to', gifname)

@torch.no_grad()
def plot_and_save(points, pngname, title='', axlim=None):
    points = points.detach().cpu().numpy()
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d')
    ax.plot3D(points[:,0], points[:,1], points[:,2], 'b')
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])

    _set_equal_axes(ax)
    limits = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())
    plt.savefig(pngname)
    plt.close(fig)
    print('Saving to', pngname)
    return limits
