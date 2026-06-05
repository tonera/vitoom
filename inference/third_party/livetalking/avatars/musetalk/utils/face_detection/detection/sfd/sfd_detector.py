import os
import cv2
from torch.utils.model_zoo import load_url

from ..core import FaceDetector

from .net_s3fd import s3fd
from .bbox import *
from .detect import *

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=os.path.join(os.path.dirname(os.path.abspath(__file__)), 's3fd.pth'),
                 verbose=False, filter_threshold=0.3):
        """SFD face detector.

        ``filter_threshold``: 检测置信度阈值。上游硬编码 0.5 在长视频里会导致部分
        帧（眨眼/微表情/运动模糊/人脸占比过大）漏检，落到 ``coord_placeholder``，
        进而让 ``genavatar`` 写出 (0,0,0,0) 的 bbox + 0 形状 mask，sidecar 推理时
        ``paste_back_frame`` 会因 broadcast shape 不匹配整段卡帧。
        默认放宽到 0.3 提高召回；调用方仍可在构造时传更高/更低的值。
        """
        super(SFDDetector, self).__init__(device, verbose)
        self.filter_threshold = filter_threshold

        # Initialise the face detector
        if not os.path.isfile(path_to_detector):
            model_weights = load_url(models_urls['s3fd'])
        else:
            model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > self.filter_threshold]

        return bboxlist

    def detect_from_batch(self, images):
        bboxlists = batch_detect(self.face_detector, images, device=self.device)
        keeps = [nms(bboxlists[:, i, :], 0.3) for i in range(bboxlists.shape[1])]
        bboxlists = [bboxlists[keep, i, :] for i, keep in enumerate(keeps)]
        bboxlists = [[x for x in bboxlist if x[-1] > self.filter_threshold] for bboxlist in bboxlists]

        return bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
