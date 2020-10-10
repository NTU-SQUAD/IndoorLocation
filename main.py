import argparse
import json
from pathlib import Path

from io_f import read_data_file
from preprocessing import MagAndWifi
from visualize_f import visualize_trajectory, visualize_heatmap


def task1_visualize_groundTruth(number=0):
    if number > len(path_filenames):
        number = len(path_filenames)-1
    path_data = read_data_file(path_filenames[number])
    path_id = path_filenames[number].name.split(".")[0]
    plt = visualize_trajectory(
        path_data.waypoint[:, 1:3],
        floor_plan_filename,
        width_meter,
        height_meter,
        title=path_id,
        dir=path_image_save_dir,
        show=True,
        save=True)
    plt.clf()
    plt.close()


def task2_mag_heatMap():
    datas = MagAndWifi(path_filenames)
    pos, val = datas.getMagKeyVal()
    plt = visualize_heatmap(pos, val, floor_plan_filename, width_meter,
                            height_meter, title='Magnetic Strength',
                            dir=magn_image_save_dir, show=True, save=True)
    plt.clf()
    plt.close()


def task3_wifi_heatMap(bssid):
    datas = MagAndWifi(path_filenames)
    if not bssid:
        bssids = ['1e:74:9c:a7:b2:e4',
                  '16:74:9c:a7:a3:c0', '12:74:9c:2c:ec:7a']
    else:
        bssids = [bssid]
    for bssid in bssids:
        print("RSSI:", bssid)
        pos, val = datas.getWifiKeyVal(bssid)
        plt = visualize_heatmap(pos, val, floor_plan_filename, width_meter,
                                height_meter, title='WIFI_' + bssid,
                                dir=wifi_image_save_dir, show=True, save=True)
        plt.clf()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Indoor Localizaiton')
    parser.add_argument('-t', '--task', default=1, type=int)
    parser.add_argument('-n', '--number', default=0, type=int)
    parser.add_argument('--input', default='./data/site1/F1')
    parser.add_argument('--bssid', default='')

    args = parser.parse_args()

    floor_data_dir = args.input
    path_data_dir = floor_data_dir + '/path_data_files'
    floor_plan_filename = floor_data_dir + '/floor_image.png'
    floor_info_filename = floor_data_dir + '/floor_info.json'

    save_dir = './output/site1/F1'
    path_image_save_dir = save_dir + '/path_images'
    step_position_image_save_dir = save_dir
    magn_image_save_dir = save_dir
    wifi_image_save_dir = save_dir + '/wifi_images'

    Path(path_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(magn_image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(wifi_image_save_dir).mkdir(parents=True, exist_ok=True)

    with open(floor_info_filename) as f:
        floor_info = json.load(f)
    width_meter = floor_info["map_info"]["width"]
    height_meter = floor_info["map_info"]["height"]

    path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))

    task = args.task
    number = args.number
    if task == 1:
        task1_visualize_groundTruth(number)
    elif task == 2:
        task2_mag_heatMap()
    elif task == 3:
        task3_wifi_heatMap(args.bssid)
