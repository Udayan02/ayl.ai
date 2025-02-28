import cv2
import numpy as np
from typing import List, Tuple, Dict

class VideoProcess:
    def __init__(self, vid_src: str, temp_src: str, vid_ext: str = None, temp_ext: str = None):
        self.vid_src = vid_src
        self.vid_ext = vid_ext
        self.temp_src = temp_src
        self.temp_ext = temp_ext
        self.video = cv2.VideoCapture(vid_src)
        self.template = cv2.imread(temp_src)

    def process(self):
        methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
                   'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
        if not self.video.isOpened():
            print(f"Error Opening Video at path: {self.src}")
            return
        fps = self.video.get(cv2.CAP_PROP_FPS)
        skip_frame = fps // 10
        frame_count = 0

        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]

        vid_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # or 'mp4v', 'avc1', etc.
        output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (vid_width, vid_height))

        while self.video.isOpened():
            if frame_count % skip_frame == 0:
                ret, frame = self.video.read()
                if not ret:
                    break
                loc = self.match_temp_frame(frame, threshold=0.4)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
                    # cv2.imshow(frame, "Res")
                output_video.write(frame)
            else:
                ret = self.video.grab()
                if not ret:
                    break
            frame_count += 1

        output_video.release()

    def match_temp_frame(self, frame, threshold: float):
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matched_res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(matched_res >= threshold)
        return loc


def main():
    vid_src = "./test_vid2.mp4"
    temp_src = "./tropicana_orange_juice.jpg"
    video_processor = VideoProcess(vid_src=vid_src, temp_src=temp_src, vid_ext=".mp4", temp_ext=".jpg")
    video_processor.process()


if __name__ == "__main__":
    main()


