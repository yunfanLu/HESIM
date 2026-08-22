import shutil
from os.path import join

import cv2
import numpy as np
from absl.logging import info

from hesim.io import vis_evs_raw
from hesim.simulator.evs_noise_generate import get_eiger_evs_simulator, get_gen2_evs_simulator
from hesim.simulator.sRGB_to_RAW import get_eiger_sRGB_to_RAW_generator, get_gen2_sRGB_to_RAW_generator


class HybridSensorSimulator:
    def __init__(
        self, sensor: str, in_video_fps: int = 3200, out_evs_fps: int = 80, out_aps_fps: int = 24, theta_scale=0.1
    ):
        sensor = sensor.lower()
        if sensor not in {"eiger", "gen2"}:
            raise ValueError(f"Unsupported sensor: {sensor}. Expected 'eiger' or 'gen2'.")
        self.sensor = sensor

        self.evs_s = get_eiger_evs_simulator(theta_scale) if sensor == "eiger" else get_gen2_evs_simulator(theta_scale)
        self.aps_s = get_eiger_sRGB_to_RAW_generator() if sensor == "eiger" else get_gen2_sRGB_to_RAW_generator()

        self.in_video_fps = in_video_fps
        self.out_evs_fps = out_evs_fps
        self.out_aps_fps = out_aps_fps

        self.aps_h, self.aps_w = 2448, 3264
        self.evs_h, self.evs_w = (1632, 1224) if sensor == "eiger" else (612, 816)

    def frames_to_evs(self, frame_path_list, evs_dir):
        step = 1.0 * self.in_video_fps / self.out_evs_fps
        num_frames = len(frame_path_list)
        for i in range(0, num_frames - int(step), int(step)):
            l = i
            r = min(i + int(step), num_frames - 1)

            Il_name = frame_path_list[i].split("/")[-1].split(".")[0]
            Il_part_name = frame_path_list[l].split("/")[-2]
            Ir_name = frame_path_list[r].split("/")[-1].split(".")[0]
            Ir_part_name = frame_path_list[r].split("/")[-2]

            evs_name = f"{i:08d}_{Il_part_name}_{Il_name}_{Ir_part_name}_{Ir_name}"
            E = 0
            for j in range(l, r):
                I0_path = frame_path_list[j]
                I1_path = frame_path_list[j + 1]

                E_delta_t = self.evs_s.simulate_from_path(I0_path, I1_path)[0]
                E = E + E_delta_t
            E = E.cpu().numpy()
            E[E > 0] = 1
            E[E < 0] = -1

            E_img = vis_evs_raw(E)
            cv2.imwrite(join(evs_dir, f"{evs_name}.jpg"), E_img)

            np.savez_compressed(join(evs_dir, f"{evs_name}.npz"), events=E)
            info(f"[{i:08d}/{num_frames}] EVS frame saved: {join(evs_dir, evs_name)}.jpg and .npz")

    def frames_to_aps_raw(self, frame_path_list, aps_dir, exposure_time_ms):
        assert (
            exposure_time_ms > 0 and exposure_time_ms < 1000.0 / self.out_aps_fps
        ), "Exposure time must be positive and less than frame interval."
        step = max(1, 1.0 * self.in_video_fps / self.out_aps_fps)
        exposure_frame_num = max(1, int(self.in_video_fps * exposure_time_ms / 1000.0))
        self.aps_s.set_exposure_time_ms(exposure_time_ms)

        num_frames = len(frame_path_list)
        for i in range(0, num_frames, int(round(step))):

            if i + exposure_frame_num > num_frames:
                break

            center_path = frame_path_list[i + exposure_frame_num // 2]
            I_name = center_path.split("/")[-1].split(".")[0]
            raw = 0
            for j in range(i, min(i + exposure_frame_num, num_frames)):
                I_path = frame_path_list[j]
                raw_delta = self.aps_s.srgb2raw_from_path(I_path)
                raw = raw + raw_delta
            raw = raw / exposure_frame_num

            cv2.imwrite(join(aps_dir, f"{I_name}_raw-vis.jpg"), raw.astype(np.uint16))
            np.savez_compressed(join(aps_dir, f"{I_name}_raw.npz"), raw=raw)
            shutil.copy(center_path, join(aps_dir, f"{I_name}.png"))
            info(f"[{i:08d}/{num_frames}] APS RAW frame saved: {join(aps_dir, f'{I_name}_raw.npz')}")


HybridSensorSumulator = HybridSensorSimulator
