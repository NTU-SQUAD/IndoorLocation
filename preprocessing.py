import numpy as np
import scipy.signal as signal


# get points of each step
class StepPoint:
    def __init__(self, accelerate_value, direction_vector, real_pos):
        self.acc = accelerate_value
        self.dv = direction_vector
        self.rp = real_pos

        self.step_points = []
        self.compute()

    def get_step_points(self):
        return self.step_points

    def compute(self):
        # calculate acc of each step
        time, step_acc = self.getSteps()

        # calculate step direciton and len
        step_directions = self.getStepDirections(time, self.dv)
        step_lengths = self.getStepLen(step_acc)

        # calculate step points
        step_points = self.computeStepPoint(step_directions, step_lengths)

        # correct the step points with ground truth
        self.step_points = self.adjust(step_points, self.rp)

    def getSteps(self):
        # times： timestamps of each step
        # step_accs：[timestamp, max, min, variance]
        acc = self.acc
        size = np.size(acc, 0)
        times, step_accs = np.array([]), np.zeros((0, 4))

        window_size = 22
        state = 0
        acc_m_pre = 0
        acc_binarys = np.zeros((window_size,), dtype=int)
        acc_max = np.zeros((2,))
        acc_min = np.zeros((2,))
        fb, fa, zf = self.filterInit(window_size)

        for i in range(size):
            acci = acc[i, :]
            acci_time, acci_val = acci[0], acci[1:]

            # acc magnitudes
            acci_m = np.sqrt(np.sum(acci_val ** 2))
            f_acci_m, zf = signal.lfilter(fb, fa, [acci_m], zi=zf)
            f_acci_m = f_acci_m[0]

            acc_binarys, acc_m_pre, acc_std = self.update_acc_binarys(f_acci_m, window_size, acc_binarys, acc_m_pre)
            if acc_binarys[-1] == 0 and acc_binarys[-2] == 1:
                acc_max = self.update_acc_max(acc_max, f_acci_m, acci_time, state)
                state = 1

            flag = False
            if acc_binarys[-1] == 0 and acc_binarys[-2] == -1:
                acc_min, flag, state = self.update_acc_min(acc_min, f_acci_m, acci_time, state)
            if flag:
                times = np.append(times, acci_time)
                step_accs = np.append(step_accs, [[acci_time, acc_max[1], acc_min[1], acc_std ** 2]], axis=0)

        return times, step_accs

    def getStepDirections(self, time, dv):
        directions = self.getD(dv)
        ltime = len(time)
        ld = len(directions)

        step_d = np.zeros((ltime, 2))
        time_idx = 0
        for i in range(ld):
            if time_idx < ltime and directions[i, 0] == time[time_idx]:
                step_d[time_idx, :] = directions[i, :]
                time_idx += 1
            else:
                break
        assert time_idx == ltime

        return step_d

    def getD(self, dv):
        size = np.size(dv, 0)
        directions = np.zeros((size, 2))
        for i in range(size):
            di = directions[i, :]
            time, vector = di[0], di[1:]

            azimuth = self.get_orientation(vector)
            directions[i, :] = time, (-azimuth) % (2 * np.pi)
        return directions

    def adjust(self, step_points, rp):
        step_points = self.time_split_data(step_points, rp[:, 0])
        if len(step_points) != rp.shape[0] - 1:
            del step_points[-1]

        size = len(step_points)
        adjust_step_points = np.zeros((0, 3))

        for i in range(size):
            start, end = rp[i], rp[i + 1]
            adjust_step_point = self.adjust_point(step_points[i], start, end)
            if i > 0:
                adjust_step_points = np.append(adjust_step_points, adjust_step_point[1:], axis=0)
            else:
                adjust_step_points = np.append(adjust_step_points, adjust_step_point, axis=0)

        return adjust_step_points

    @staticmethod
    def getStepLen(step_acc):
        size = step_acc.shape[0]
        step_lens = np.zeros((size, 2))
        times = np.zeros((size - 1,))
        step_lens[:, 0] = step_acc[:, 0]
        times_temp = np.zeros((0,))

        # cal step period
        for i in range(times.shape[0]):
            stepi_t = step_acc[i + 1, 0] - step_acc[i, 0] / 1000
            times_temp = np.append(times_temp, [stepi_t])
            if times_temp.shape[0] > 2:
                times_temp = np.delete(times_temp, [0])
            times[i] = np.sum(times_temp / times_temp.shape[0])

        # cal parameter k
        k0, kmin, kmax = 0.4, 0.4, 0.8
        para_a0, para_a1, para_a2 = 0.21468084, 0.09154517, 0.02301998

        k = np.zeros(size)
        k[0] = k0
        for i in range(times.shape[0]):
            k[i + 1] = np.max([(para_a0 + para_a1 / times[i] + para_a2 * step_acc[i, 3]), kmin])
            k[i + 1] = np.min([k[i + 1], kmax]) * (k0 / kmin)

        # cal step len
        step_lens[:, 1] = np.max([(step_acc[:, 1] - step_acc[:, 2]), np.ones((size,))], axis=0) ** 0.25 * k

        return step_lens

    @staticmethod
    def computeStepPoint(step_directions, step_lengths):
        size = step_lengths.shape[0]
        step_points = np.zeros((size, 3))

        for i in range(size):
            step_points[i, 0] = step_lengths[i, 0]
            step_points[i, 1] = -step_lengths[i, 1] * np.sin(step_directions[i, 1])
            step_points[i, 1] = step_lengths[i, 2] * np.cos(step_directions[i, 1])

        return step_points

    @staticmethod
    def filterInit(window_size):
        # init Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        # get final filter delay values
        warmup_data = np.ones((window_size,)) * 9.81

        fb, fa = signal.butter(4, 0.08)
        zi = signal.lfilter_zi(fb, fa)
        _, zi = signal.lfilter(fb, fa, warmup_data, zi=zi)
        _, zf = signal.lfilter(fb, fa, warmup_data, zi=zi)

        return fb, fa, zf

    @staticmethod
    def update_acc_binarys(f_acci_m, window_size, acc_binarys, acc_m_pre):
        # acceleration magnitudes window
        # find steps based on this
        acc_m_win = np.zeros((window_size, 1))
        acc_m_win = np.append(acc_m_win, [f_acci_m])
        acc_m_win = np.delete(acc_m_win, 0)
        mean_gravity = np.mean(acc_m_win)
        acc_std = np.std(acc_m_win)
        mag_threshold = np.max([0.6, 0.4 * acc_std])
        acc_mf_detrend = f_acci_m - mean_gravity

        # get acc binarys according to the position of the detrend
        peak = np.max([acc_m_pre, mag_threshold])
        valley = np.min([acc_m_pre, -mag_threshold])
        acc_m_pre = acc_mf_detrend

        if acc_mf_detrend > peak:
            return np.delete(np.append(acc_binarys, [1]), 0)
        if acc_mf_detrend < valley:
            return np.delete(np.append(acc_binarys, [-1]), 0)

        return np.delete(np.append(acc_binarys, [0]), 0), acc_m_pre, acc_std

    @staticmethod
    def update_acc_max(acc_max, f_acci_m, acci_time, state):
        if state == 0 or (state == 1) and ((acci_time - acc_max[0]) <= 250) and (f_acci_m > acc_max[1]) or (
                acci_time - acc_max[0]) > 250:
            acc_max[:] = acci_time, f_acci_m

        return acc_max

    @staticmethod
    def update_acc_min(acc_min, f_acci_m, acci_time, state):
        flag = False
        if (state == 1) and ((acci_time - acc_min[0]) > 250):
            acc_min[:] = acci_time, f_acci_m
            state = 2
            flag = True
        elif (state == 2) and ((acci_time - acc_min[0]) <= 250) and (f_acci_m < acc_min[1]):
            acc_min[:] = acci_time, f_acci_m

        return acc_min, flag, state

    @staticmethod
    def get_orientation(rotation):
        q1, q2, q3 = rotation[:3]
        if rotation.size >= 4:
            q0 = rotation[3]
        else:
            q0 = 1 - q1 * q1 - q2 * q2 - q3 * q3
            if q0 > 0:
                q0 = np.sqrt(q0)
            else:
                q0 = 0

        sq_q1 = 2 * q1 * q1
        sq_q2 = 2 * q2 * q2
        sq_q3 = 2 * q3 * q3
        q1_q2 = 2 * q1 * q2
        q3_q0 = 2 * q3 * q0
        q1_q3 = 2 * q1 * q3
        q2_q0 = 2 * q2 * q0
        q2_q3 = 2 * q2 * q3
        q1_q0 = 2 * q1 * q0

        R = np.zeros((9,))
        if R.size == 9:
            R[0] = 1 - sq_q2 - sq_q3
            R[1] = q1_q2 - q3_q0
            R[2] = q1_q3 + q2_q0

            R[3] = q1_q2 + q3_q0
            R[4] = 1 - sq_q1 - sq_q3
            R[5] = q2_q3 - q1_q0

            R[6] = q1_q3 - q2_q0
            R[7] = q2_q3 + q1_q0
            R[8] = 1 - sq_q1 - sq_q2

            R = np.reshape(R, (3, 3))
        elif R.size == 16:
            R[0] = 1 - sq_q2 - sq_q3
            R[1] = q1_q2 - q3_q0
            R[2] = q1_q3 + q2_q0
            R[3] = 0.0

            R[4] = q1_q2 + q3_q0
            R[5] = 1 - sq_q1 - sq_q3
            R[6] = q2_q3 - q1_q0
            R[7] = 0.0

            R[8] = q1_q3 - q2_q0
            R[9] = q2_q3 + q1_q0
            R[10] = 1 - sq_q1 - sq_q2
            R[11] = 0.0

            R[12] = R[13] = R[14] = 0.0
            R[15] = 1.0

            R = np.reshape(R, (4, 4))

        flat_R = R.flatten()
        orientation = np.zeros((3,))
        if np.size(flat_R) == 9:
            orientation[0] = np.arctan2(flat_R[1], flat_R[4])
            orientation[1] = np.arcsin(-flat_R[7])
            orientation[2] = np.arctan2(-flat_R[6], flat_R[8])
        else:
            orientation[0] = np.arctan2(flat_R[1], flat_R[5])
            orientation[1] = np.arcsin(-flat_R[9])
            orientation[2] = np.arctan2(-flat_R[8], flat_R[10])

        return orientation[0]

    @staticmethod
    def time_split_data(datas, times):
        datas_time = datas[:, 0]
        times = np.unique(times)
        size = times.shape[0]
        _datas = []

        start = 0

        for i in range(size):
            end = np.searchsorted(datas_time, times[i], side='right')
            if end == start:
                continue
            _datas.append(datas[start:end, :].copy())
            start = end

        _datas.append(datas[start:, :].copy())
        return _datas

    @staticmethod
    def adjust_point(point, start, end):
        pos = np.zeros(point.shape)
        pos[:, 0], pos[0, 1:3] = point[:, 0], point[0, 1:3] + start[1:3]
        for i in range(1, point.shape[0]):
            pos[i, 1:3] = pos[i - 1, 1:3] + point[i, 1:3]
        pos = np.insert(pos, 0, start, axis=0)

        old = pos[:, 1:3]
        A = pos[0, 1:3]
        B = end[1:3]
        Bp = pos[-1, 1:3]
        new_xy = np.append(np.zeros((0, 2)), [A], 0)

        angle_BpAB = np.arctan2(Bp[1] - A[1], Bp[0] - A[0]) - np.arctan2(B[1] - A[1], B[0] - A[0])
        AB = np.sqrt(np.sum((B - A) ** 2))
        ABp = np.sqrt(np.sum((Bp - A) ** 2))

        for i in range(1, np.size(old, 0)):
            angle_CAX = np.arctan2(old[i, 1] - A[1], old[i, 0] - A[0]) - angle_BpAB
            AC = np.sqrt(np.sum((old[i, :] - A) ** 2)) * AB / ABp
            delta_C = np.array([AC * np.cos(angle_CAX), AC * np.sin(angle_CAX)])
            new_xy = np.append(new_xy, [delta_C + A], 0)

        return np.column_stack((pos[:, 0], new_xy))


# get mag and wifi info of each step
class MagAndWifi:

    def __init__(self, files):
        self.files = files
        self.dataMap = None
        self.mag = None
        self.wifi = None
        self.preProcess()

    def preProcess(self):
        for f in self.files:
            # TODO: read from dataClass
            print(f)
            raw_datas = {
                "acc":"",
                "dv":"",
                "rp":"",
                "mag":"",
                "wifi":""
            }
            acc = raw_datas.acc
            direciton = raw_datas.dv
            real_pos = raw_datas.rp
            mag = raw_datas.mag
            wifi = raw_datas.wifi

            step_points = StepPoint(acc, direciton, real_pos).get_step_points()
            self.dataToMap(mag, step_points, "mag")
            if wifi.size > 0:
                self.dataToMap(wifi, step_points, "wifi")

    def dataToMap(self, datas, step_points, name):
        time_split_data = StepPoint.time_split_data(datas, datas[:,0])
        for data in time_split_data:
            idx = np.argmin(np.abs(step_points[:, 0] - float(data[0, 0])))
            point_key = tuple(step_points[idx, 1:3])

            if point_key not in self.dataMap:
                self.dataMap[point_key] = {
                    'mag': np.zeros((0, 4)),
                    'wifi': np.zeros((0, 5)),
                }

            self.dataMap[point_key][name] = np.append(self.dataMap[point_key][name], data, axis=0)

    def magProcess(self):
        for key in self.dataMap:
            mag = self.dataMap[key]['mag']
            mag_val = np.mean(np.sqrt(np.sum(mag[:, 1:4] ** 2, axis=1)))
            self.mag[key] = mag_val

    def getMagKeyVal(self):
        if not self.mag:
            self.magProcess()

        pos = np.array(list(self.mag.keys()))
        vals = np.array(list(self.mag.values()))

        return pos, vals

    def wifiProcess(self):
        for key in self.dataMap:
            wifi = self.dataMap[key]['wifi']

            for data in wifi:
                bssid, rssi = data[2], int(data[3])

                if bssid not in self.wifi:
                    rssi_point = {key: np.array([rssi, 1])}
                else:
                    rssi_point = self.wifi[bssid]
                    if key not in rssi_point:
                        rssi_point[key] = np.array([rssi, 1])
                    else:
                        count = rssi_point[key][1] + 1
                        val = (rssi_point[key][0] * (count-1) + rssi) / count

                        rssi_point[key][1] = count
                        rssi_point[key][0] = val

                self.wifi[bssid] = rssi_point

    def getWifiKeyVal(self, rssi):
        if not self.wifi:
            self.wifiProcess()

        pos = np.array(list(self.wifi[rssi].keys()))
        vals = np.array(list(self.wifi[rssi].values()))

        return pos, vals
