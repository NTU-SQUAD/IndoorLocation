from collections import namedtuple
import numpy as np

ReadData = namedtuple("ReadData",
                      [
                          "acce",
                          "magn",
                          "ahrs",
                          "wifi",
                          "waypoint",
                      ])


def read_data_file(data_filename):
    acce = []
    magn = []
    ahrs = []
    wifi = []
    waypoint = []

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue

        line_data = line_data.split('\t')

        # 加速度数据：[时间戳，x方向加速度，y方向加速度]
        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]),
                         float(line_data[3]), float(line_data[4])])
            continue

        # 地磁数据：[时间戳，x方向，y方向]
        if line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]),
                         float(line_data[3]), float(line_data[4])])
            continue

        # 方向向量：[时间戳，x方向，y方向] 和 acc 一起计算acc pos
        if line_data[1] == 'TYPE_ROTATION_VECTOR':
            ahrs.append([int(line_data[0]), float(line_data[2]),
                         float(line_data[3]), float(line_data[4])])
            continue

        # wifi数据：[当前时间戳，wifi设备的ssid，接收信号强度（为负数，值越大（越接近0）信号越强），上次接收到的时间戳]
        if line_data[1] == 'TYPE_WIFI':
            sys_ts = line_data[0]
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = line_data[4]
            lastseen_ts = line_data[6]
            wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
            wifi.append(wifi_data)
            continue

        # Ground Truth, x,y方向上的位置 单位为米
        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(
                line_data[2]), float(line_data[3])])

    acce = np.array(acce)
    magn = np.array(magn)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    waypoint = np.array(waypoint)

    return ReadData(acce, magn, ahrs, wifi, waypoint)
