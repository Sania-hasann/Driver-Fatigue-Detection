import cv2
import numpy as np
from face_geometry import PCF, get_metric_landmarks, procrustes_landmark_basis
from utils import rot_mat_to_euler

class HeadPoseEstimator:
    def __init__(self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False):

        self.NOSE_AXES_POINTS = np.array(
            [[7, 0, 10], [0, 7, 6], [0, 0, 14]], dtype=float
        )
        self.JAW_LMS_NUMS = [61, 291, 199]

        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.focal_length = None

        self.pcf_calculated = False

        self.model_lms_ids = self._get_model_lms_ids()

    def _get_model_lms_ids(self):
        model_lms_ids = self.JAW_LMS_NUMS + [
            key for key, _ in procrustes_landmark_basis
        ]
        model_lms_ids.sort()

        return model_lms_ids

    def get_pose(self, frame, landmarks, frame_size):
        rvec = None
        tvec = None
        model_img_lms = None
        eulers = None
        metric_lms = None

        if not self.pcf_calculated:
            self._get_camera_parameters(frame_size)

        model_img_lms = (
            np.clip(landmarks[self.model_lms_ids, :2], 0.0, 1.0) * frame_size
        )

        metric_lms = get_metric_landmarks(landmarks.T.copy(), self.pcf)[0].T

        model_metric_lms = metric_lms[self.model_lms_ids, :]

        (solve_pnp_success, rvec, tvec) = cv2.solvePnP(
            model_metric_lms,
            model_img_lms,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        tvec = tvec.round(2)

        if solve_pnp_success:
            rvec, tvec = cv2.solvePnPRefineVVS(
                model_metric_lms,
                model_img_lms,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
            )

            rvec1 = np.array([rvec[2, 0], rvec[0, 0], rvec[1, 0]]).reshape((3, 1))

            rmat, _ = cv2.Rodrigues(rvec1)

            eulers = rot_mat_to_euler(rmat).reshape((-1, 1))

            self._draw_nose_axes(frame, rvec, tvec, model_img_lms)

            return frame, eulers[0], eulers[1], eulers[2]

        else:
            return None, None, None, None

    def _draw_nose_axes(self, frame, rvec, tvec, model_img_lms):
        (nose_axes_point2D, _) = cv2.projectPoints(
            self.NOSE_AXES_POINTS, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        nose = tuple(model_img_lms[0, :2].astype(int))

        nose_x = tuple(nose_axes_point2D[0, 0].astype(int))
        nose_y = tuple(nose_axes_point2D[1, 0].astype(int))
        nose_z = tuple(nose_axes_point2D[2, 0].astype(int))

    def _get_camera_parameters(self, frame_size):
        fr_w = frame_size[0]
        fr_h = frame_size[1]
        if self.camera_matrix is None:
            fr_center = (fr_w // 2, fr_h // 2)
            focal_length = fr_w
            self.camera_matrix = np.array(
                [
                    [focal_length, 0, fr_center[0]],
                    [0, focal_length, fr_center[1]],
                    [0, 0, 1],
                ],
                dtype="double",
            )
            self.focal_length = focal_length
        else:
            self.focal_length = self.camera_matrix[0, 0]
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((5, 1))

        self.pcf = PCF(frame_height=fr_h, frame_width=fr_w, fy=self.focal_length)

        self.pcf_calculated = True
