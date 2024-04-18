# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import SevModel
from ultralytics.utils import DEFAULT_CFG, RANK


class SevTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.sev import SevTrainer

        args = dict(model='yolov8n-sev.pt', data='dota8.yaml', epochs=3)
        trainer = SevTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SevTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "sev"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model = SevModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SevValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "sev_loss"  # TODO: How is this used? Add 'sev_loss'?
        return yolo.sev.SevValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
