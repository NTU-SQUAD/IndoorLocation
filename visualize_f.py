
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np


def visualize_trajectory(trajectory, floor_plan_filename, width_meter, height_meter, title=None, dir=None, show=False, save=False):
    # add trajectory
    size_list = [6] * trajectory.shape[0]
    size_list[0] = 10
    size_list[-1] = 10

    color_list = [(4/255, 174/255, 4/255, 0.5)] * trajectory.shape[0]
    color_list[0] = (12/255, 5/255, 235/255, 1)
    color_list[-1] = (235/255, 5/255, 5/255, 1)

    position_count = {}
    text_list = []
    for i in range(trajectory.shape[0]):
        if str(trajectory[i]) in position_count:
            position_count[str(trajectory[i])] += 1
        else:
            position_count[str(trajectory[i])] = 0
        text_list.append(
            '        ' * position_count[str(trajectory[i])] + f'{i}')
    text_list[0] = 'Start Point: 0'
    text_list[-1] = f'End Point: {trajectory.shape[0] - 1}'

    plt.clf()
    img = plt.imread(floor_plan_filename)
    plt.imshow(img, extent=[0, width_meter, 0, height_meter])
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        linestyle=':',
        linewidth=2,
        color=(100/255, 10/255, 100/255),
    )
    plt.scatter(
        x=trajectory[:, 0],
        y=trajectory[:, 1],
        s=size_list,
        c=color_list,
    )
    for x,y,txt in zip(trajectory[:, 0],trajectory[:, 1],text_list):
        plt.annotate(txt, (x, y),fontsize='xx-small')
    plt.xlim([0, width_meter])
    plt.ylim([0, height_meter])
    plt.title(label=title or "No title.", loc="left")

    if save:
        plt.savefig("{}/{}.png".format(dir, title), dpi=300)
    if show:
        plt.show()

    return plt


def visualize_heatmap(position, value, floor_plan_filename, width_meter, height_meter, colorbar_title="colorbar", title=None, dir=None, show=False, save=False):
    plt.clf()
    cmap=plt.get_cmap("rainbow",256)
    img = plt.imread(floor_plan_filename)
    plt.imshow(img, extent=[0, width_meter, 0, height_meter])
    plt.scatter(
        x=position[:, 0],
        y=position[:, 1],
        s=7,
        c=value,
        cmap=cmap,
        alpha=0.8,
        edgecolors='none',
    )
    norm = mpl.colors.Normalize(vmin=value.min(), vmax=value.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,
                        ticks=np.linspace(int(value.min())-.1, int(value.max())+.1, 20),
                        )
    cbar.set_label(colorbar_title)
    plt.xlim([0, width_meter])
    plt.ylim([0, height_meter])
    plt.title(label=title or "No title.", loc="left")

    if save:
        plt.savefig("{}/{}.png".format(dir, title), dpi=300)
    if show:
        plt.show()

    return plt
