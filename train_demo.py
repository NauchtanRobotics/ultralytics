from pathlib import Path

from ultralytics import YOLO


def train():
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("/home/david/addn_repos/ultralytics/runs/detect/r3v8srd40.2.1/weights/best.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(
        data="/home/david/production/yolov5/datasets/srd40.2.1/dataset.yaml",
        start_epoch=7,
        epochs=50,
        name="r3v8srd40.2.1",
        cache="ram",
        imgsz=800,
        batch=46,
        workers=6,
        device="0,1"
    )  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format


def evalualte():
    model = YOLO(
        "/home/david/production/ultralytics/runs/detect/srd40.2.1_yolov8/weights/best.pt",
        task="predict",
        verbose=False
    )
    results = model.predict(
        source=Path("/media/david/Carol_sexy/North_Burnett_2022/Andersons Road - 2034/Photos"),
        project=Path("/home/david/production/sealed_roads_dataset/.predict"),
        name="anderson_rd_north_burnett",
        conf=0.1,
        imgsz=1024,
        device=0,
        augment=True,
        save_conf=True,
        save_txt=True,
        save=True
    )
    # print(results.__dir__)\results[0].boxes.cls[0]
    # results[0].orig_img: ndarray
    # results[0].boxes.cls[0]: float
    # results[0].path: str
    # results[0].boxes.conf[0]
    # results[0].boxes.xywhn[0]: Tensor
    # results[0].names: Dict[int, str]  # class_id_to_label_map


if __name__ == "__main__":
    evalualte()
