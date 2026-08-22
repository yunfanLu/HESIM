from os import listdir, makedirs
from os.path import isdir, join

from absl import app
from absl.flags import FLAGS, DEFINE_float, DEFINE_string
from absl.logging import info

from hesim.simulator.application import HybridSensorSimulator

DEFINE_string("SENSOR", "EIGER", "Sensor type: EIGER or GEN2")
DEFINE_float("EXPOSURE_TIME_MS", 20.0, "APS exposure time in milliseconds.")
DEFINE_string("INPUT_DIR", "", "Root directory containing high-frame-rate video groups.")
DEFINE_string("OUT_DIR", "", "Output directory for simulated data; empty uses the default dataset path.")


def sim(sensor, input_dir, exposure_time_ms, out_dir=""):
    theta_scale = 0.8
    output_dir = out_dir or join(
        "dataset/HKUST-HighSpeedVideo/HESIM-Simulated",
        f"{sensor.upper()}-ThetaScale{theta_scale}-EXP{int(exposure_time_ms)}ms",
    )
    makedirs(output_dir, exist_ok=True)

    hybrid_sensor_simulator = HybridSensorSimulator(
        sensor=sensor, in_video_fps=3200, out_evs_fps=800, out_aps_fps=24, theta_scale=theta_scale
    )
    video_groups = sorted(listdir(input_dir))
    for video_group in video_groups:
        video_group_path = join(input_dir, video_group)
        if not isdir(video_group_path):
            continue
        video_parts = sorted(listdir(video_group_path))
        all_frame_pathes = []
        for video_part in video_parts:
            video_part_path = join(video_group_path, video_part)
            if not isdir(video_part_path):
                continue
            frame_files = sorted(listdir(video_part_path))
            all_frame_pathes.extend([join(video_part_path, ff) for ff in frame_files])

        our_video_folder = join(output_dir, video_group)
        makedirs(our_video_folder, exist_ok=True)

        evs_dir = join(our_video_folder, f"EVS")
        aps_dir = join(our_video_folder, f"APS")
        makedirs(evs_dir, exist_ok=True)
        makedirs(aps_dir, exist_ok=True)
        info(f"Processing video part: {video_part_path}, total frames: {len(all_frame_pathes)}")
        info(f"EVS output dir: {evs_dir}")
        info(f"APS RAW output dir: {aps_dir}")
        hybrid_sensor_simulator.frames_to_aps_raw(all_frame_pathes, aps_dir, exposure_time_ms=exposure_time_ms)
        hybrid_sensor_simulator.frames_to_evs(all_frame_pathes, evs_dir)


def main(args):
    if not FLAGS.INPUT_DIR:
        raise app.UsageError("--INPUT_DIR is required.")
    sim(FLAGS.SENSOR, FLAGS.INPUT_DIR, exposure_time_ms=FLAGS.EXPOSURE_TIME_MS, out_dir=FLAGS.OUT_DIR)


if __name__ == "__main__":
    app.run(main)
