import plotly.graph_objs as go
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

def save_figure_to_html(fig, filename):
    fig.write_html(filename)

# 绘制轨迹（waypoint的数据（groud truth), 底图，楼层宽，楼层高）


def visualize_trajectory(trajectory, floor_plan_filename, width_meter, height_meter, title=None, mode='lines + markers + text', show=False):
    # add trajectory
    # 轨迹点大小，头尾为10，其他为6
    size_list = [6] * trajectory.shape[0]
    size_list[0] = 10
    size_list[-1] = 10

    # 轨迹点颜色
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

    if show:
        plt.show()
    return plt


def visualize_heatmap(position, value, floor_plan_filename, width_meter, height_meter, colorbar_title="colorbar", title=None, show=False):
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
                        # boundaries=np.arange(value.min()-1, value.max() + 1,5)
                        )
    cbar.set_label(colorbar_title)
    # for x,y,txt in zip(position[:, 0],position[:, 1],value):
        # plt.annotate(txt, (x, y),fontsize='xx-small')
    plt.xlim([0, width_meter])
    plt.ylim([0, height_meter])
    plt.title(label=title or "No title.", loc="left")
    if show:
        plt.show()

    return plt
