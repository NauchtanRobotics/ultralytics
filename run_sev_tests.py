from pathlib import Path

from ultralytics import YOLO

"""
.venv/bin/python -c"from run_sev_tests import test_sev; test_sev()"
"""


def test_sev():
    # Create a new YOLOv8n-OBB model from scratch
    model = YOLO('yolov8x-sev.yaml', task="sev")

    data_yaml = Path("/home/david/production/severity42.2/dataset.yaml")
    # TODO: problem in ultralytics.data.augment.RandomPerspective.apply_bboxes assumes 4 points not 5. How
    #       do we avoid augmenting the severity field but still have it included?
    results = model.train(
        data=data_yaml,
        epochs=300,
        imgsz=800,
        cache="RAM",
        device="0,1",  # "0,1",
        workers=8,
        # lr0=0.1,  # ignored due to optimizer being in auto?
        # optimizer='SGD',
        # weight_decay=0.0025,
        # max_det=30,
        name="sevX42.2",
        iou=0.4,
        conf=0.2,
        batch=20,
        project=Path("/home/david/production/sealed_roads_dataset/.train"),  # or could put it in "runs/severity/train"
        # amp=False  # connection reset error
    )

    # metrics = model.val()
    #
    # # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

