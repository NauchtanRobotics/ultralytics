from pathlib import Path

from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


def main(video_path: Path):
    assert video_path.exists(), f"'video_path' parameter doesn't exist or not a Path object."
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Error on line 7 reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define line points
    line_points = [(20, 400), (1080, 400)]

    # Video writer
    video_writer = cv2.VideoWriter("object_counting_output.avi",
                           cv2.VideoWriter_fourcc(*'mp4v'),
                           fps,
                           (w, h))

    # Init Object Counterfasfastwhat
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=line_points,
                     classes_names=model.names,
                     draw_tracks=True)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False)

        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     video = Path(r"C:\Users\61419\OneDrive - Shepherd Services\Pictures\Camera Roll\WIN_20240110_16_07_01_Pro.mp4")
#     main(video_path=video)


def track_and_count_image_sequence(images_root: Path):
    model = YOLO("yolov8l.pt")
    w, h, fps = 1920, 1080, 2

    # Define line points
    line_points = [(20, 400), (1080, 400)]

    # Video writer
    video_writer = cv2.VideoWriter("object_counting_output.avi",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w, h))

    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                     reg_pts=line_points,
                     classes_names=model.names,
                     draw_tracks=True)
    sorted_photos = sorted(list(images_root.iterdir()))
    my_count = 0
    for img_path in sorted_photos:
        if my_count >= 70:
            break
        if img_path.stat().st_size < 4:
            continue
        my_count += 1
        try:
            img = cv2.imread(str(img_path))
            tracks = model.track(img, persist=True, show=False)

            img = counter.start_counting(img, tracks)
            video_writer.write(img)
        except:
            pass

    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images_folder = Path(r"D:\Traffic_Signs_Brisbane\Tesla_to_Gowan\GowanRoad\Photos")
    track_and_count_image_sequence(images_root=images_folder)
