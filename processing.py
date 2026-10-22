
import cv2
import numpy as np


def get_video_info(video_path):
    """Read basic video metadata."""
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if fps > 0 else 0.0,
    }


def open_video_writer(output_path, fps, width, height):
    """Create an MP4 video writer."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (int(width), int(height)),
    )

    if not writer.isOpened():
        writer.release()
        return None

    return writer


def draw_segmentation_results(
    frame,
    result,
    show_masks=True,
    show_boxes=True,
    show_labels=True,
    show_conf=True,
):
    """Draw handrail segmentation and return detection details."""
    detections = []
    height, width = frame.shape[:2]

    if result.masks is not None:
        for mask in result.masks.data:
            mask_array = mask.cpu().numpy().astype(np.float32)
            mask_array = cv2.resize(
                mask_array,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            binary_mask = mask_array > 0.5

            if show_masks:
                overlay = frame.copy()
                overlay[binary_mask] = (255, 180, 0)
                frame = cv2.addWeighted(
                    frame, 0.65, overlay, 0.35, 0
                )

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = float(confidences[index])

            detections.append({
                "detection": index + 1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": round(confidence, 4),
            })

            if show_boxes:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (255, 180, 0), 2
                )

            if show_labels:
                label = "HANDRAIL"
                if show_conf:
                    label += f" {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA,
                )

    return frame, detections


class PersistentRailMask:
    """
    Preserve the detected rail mask across brief detection gaps.

    Best suited to stationary CCTV cameras. The mask union can retain
    newly detected sections but can drift if the camera moves.
    """

    def __init__(self, hold_frames=30):
        self.mask = None
        self.missed_frames = 0
        self.hold_frames = max(0, int(hold_frames))

    def update(self, result, frame_shape):
        height, width = frame_shape[:2]
        current = np.zeros((height, width), dtype=np.uint8)

        if result.masks is not None:
            for mask in result.masks.data:
                mask_array = mask.cpu().numpy().astype(np.float32)
                mask_array = cv2.resize(
                    mask_array,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                current[mask_array > 0.5] = 255

        if np.any(current):
            if self.mask is None:
                self.mask = current.copy()
            else:
                self.mask = cv2.bitwise_or(self.mask, current)

            self.missed_frames = 0
        else:
            self.missed_frames += 1
            if self.missed_frames > self.hold_frames:
                self.mask = None

        return None if self.mask is None else self.mask.copy()


def draw_persistent_rail(frame, rail_mask):
    """Draw the stabilized handrail mask."""
    if rail_mask is None or not np.any(rail_mask):
        return frame

    overlay = frame.copy()
    overlay[rail_mask > 0] = (255, 180, 0)

    return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)


def classify_and_draw_people(
    frame,
    pose_result,
    rail_mask,
    near_distance=25,
    wrist_confidence=0.25,
    show_ids=True,
):
    """
    Green: at least one confident wrist is near the rail.
    Red: neither wrist passes the proximity test.
    """
    using_count = 0
    violation_count = 0

    if pose_result.boxes is None or len(pose_result.boxes) == 0:
        return frame, using_count, violation_count

    boxes = pose_result.boxes.xyxy.cpu().numpy()
    keypoints = pose_result.keypoints

    track_ids = None
    if pose_result.boxes.id is not None:
        track_ids = pose_result.boxes.id.cpu().numpy().astype(int)

    distance_map = None
    if rail_mask is not None and np.any(rail_mask):
        distance_map = cv2.distanceTransform(
            (rail_mask == 0).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        using_rail = False

        if keypoints is not None and distance_map is not None:
            points = keypoints.xy[index].cpu().numpy()

            point_confidences = None
            if keypoints.conf is not None:
                point_confidences = (
                    keypoints.conf[index].cpu().numpy()
                )

            # COCO keypoint indices: 9 = left wrist, 10 = right wrist.
            for wrist_index in (9, 10):
                if wrist_index >= len(points):
                    continue

                wx, wy = points[wrist_index]
                wx, wy = int(wx), int(wy)

                if point_confidences is not None:
                    if point_confidences[wrist_index] < wrist_confidence:
                        continue

                if not (
                    0 <= wx < distance_map.shape[1]
                    and 0 <= wy < distance_map.shape[0]
                ):
                    continue

                if distance_map[wy, wx] <= near_distance:
                    using_rail = True
                    break

        if using_rail:
            color = (0, 200, 0)
            label = "USING HANDRAIL"
            using_count += 1
        else:
            color = (0, 0, 255)
            label = "VIOLATION"
            violation_count += 1

        if show_ids and track_ids is not None:
            label = f"ID {track_ids[index]} | {label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame, using_count, violation_count
