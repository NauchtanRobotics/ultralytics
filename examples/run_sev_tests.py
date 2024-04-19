from pathlib import Path

from ultralytics import YOLO


def test_sev():
    # Create a new YOLOv8n-OBB model from scratch
    model = YOLO('yolov8-sev.yaml', task="sev")

    data_yaml = Path("/home/david/production/severity42.2/dataset.yaml")
    # TODO: problem in ultralytics.data.augment.RandomPerspective.apply_bboxes assumes 4 points not 5. How
    #       do we avoid augmenting the severity field but still have it included?
    results = model.train(
        data=data_yaml,
        epochs=1,
        imgsz=800,
        cache="ram",
        device="0,1",
        workers=6,
        name="srd42.2",
        batch=42,
        project=Path("/home/david/production/sealed_roads_dataset/.train")  # or could put it in "runs/severity/train"
    )

    # val = model.val()
    #
    # # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

