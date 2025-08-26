
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import concurrent.futures as cf
import argparse


def sample_video(video_path: Path, extract_dir: Path, sampling_period: int = 6, jobs: int = 1):
    vid = cv2.VideoCapture(str(video_path))
    extract_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    parallel_saver = cf.ThreadPoolExecutor(max_workers=jobs)

    for frame_idx in tqdm(range(n_frames), desc='sampling video'):
        _, frame = vid.read()
        # print(frame_idx//period, frame_idx%period)
        if frame_idx % sampling_period == 0:
            parallel_saver.submit(cv2.imwrite,
                                  str(extract_dir/f'{(frame_idx+1):09d}.png'),
                                  frame)
    vid.release()


def main(args):
    video_fps = 30  # all the jigsaws videos are recorded at 30 fps
    # please leave this unchanged as 6, otherwise the rest of the files will not be compatible with the sampled rgb frames
    sampiling_period: int = video_fps//args.frequency

    # find all the files that need to be processed
    if not args.recursive:
        video_dirs = [Path(args.data_dir).resolve()]
    else:
        video_dirs = [v_p for v_p in Path(args.data_dir).rglob(
            '*.avi')]        # print(frame_idx//period, frame_idx%period

    for directory in tqdm(video_dirs, desc='unpacking dataset', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        if 'capture1' in directory.stem:  # Only using left video due to the similarity between left and right videos
            frame_dir = (directory.parent/'frame_{}Hz'.format(args.frequency) /
                         directory.stem.replace('_capture1', ''))
            sample_video(directory, frame_dir, sampiling_period, args.jobs)
        else:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/dataset/JIGSAWS',
                        help='path pointing to the video directory')
    parser.add_argument('--frequency', type=int,
                        help='sampling rate in Hz', choices=[1, 5, 10, 30], default=5)
    parser.add_argument(
        '-r', '--recursive', help='search recursively for video directories that have video_left.avi as a child', action='store_true', default=True)
    parser.add_argument(
        '-j', '--jobs', help='number of parallel works to use when saving images', default=4, type=int)
    args = parser.parse_args()

    main(args)
