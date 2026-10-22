
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from preprocessing import (
    get_video_info,
    open_video_writer,
    draw_segmentation_results,
    PersistentRailMask,
    draw_persistent_rail,
    classify_and_draw_people,
)


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Handrail Detection Lab",
    page_icon="🪜",
    layout="wide",
)

st.title("🪜 Handrail Detection Lab")
st.caption(
    "Persistent handrail detection, person tracking, "
    "and wrist-proximity classification."
)


# ---------------------------------------------------------
# LOAD HANDRAIL MODEL AUTOMATICALLY
# ---------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "best.pt"

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_segmentation_model(model_path):
    return YOLO(model_path)


try:
    model = load_segmentation_model(str(MODEL_PATH.resolve()))
except Exception as exc:
    st.error(f"Could not load handrail model: {exc}")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("Model")
    st.success("best.pt loaded")

    st.header("Input")

    mode = st.radio(
        "Media type",
        ["Video", "Image"],
        horizontal=True,
    )

    st.header("Handrail detection")

    confidence = st.slider(
        "Handrail confidence",
        0.05, 0.95, 0.25, 0.05,
    )

    iou = st.slider(
        "Detection IoU threshold",
        0.10, 0.90, 0.45, 0.05,
    )

    image_size = st.select_slider(
        "Inference image size",
        options=[480, 640, 768, 960],
        value=640,
    )

    show_masks = st.toggle(
        "Show persistent handrail mask",
        value=True,
    )

    show_rail_boxes = st.toggle(
        "Show handrail boxes and confidence",
        value=False,
    )

    persist_frames = st.slider(
        "Hold mask after missed detections (frames)",
        min_value=0,
        max_value=120,
        value=30,
        step=5,
    )

    st.header("Person tracking")

    use_pose = st.toggle(
        "Enable pose estimation and tracking",
        value=True,
        disabled=(mode == "Image"),
    )

    pose_model_name = st.selectbox(
        "Pose model",
        [
            "yolo11n-pose.pt",
            "yolov8n-pose.pt",
            "yolo11s-pose.pt",
        ],
        disabled=(mode == "Image" or not use_pose),
    )

    pose_confidence = st.slider(
        "Person confidence",
        0.05, 0.95, 0.25, 0.05,
        disabled=(mode == "Image" or not use_pose),
    )

    wrist_confidence = st.slider(
        "Minimum wrist confidence",
        0.05, 0.95, 0.25, 0.05,
        disabled=(mode == "Image" or not use_pose),
    )

    near_distance = st.slider(
        "Wrist-to-rail distance (pixels)",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        disabled=(mode == "Image" or not use_pose),
    )

    show_track_ids = st.toggle(
        "Show tracking IDs",
        value=True,
        disabled=(mode == "Image" or not use_pose),
    )

    tracker = st.selectbox(
        "Person tracker",
        ["bytetrack.yaml", "botsort.yaml"],
        disabled=(mode == "Image" or not use_pose),
    )

    st.header("Recording and exports")

    record_annotated_video = st.toggle(
        "🔴 Record annotated video",
        value=True,
        disabled=(mode == "Image"),
        help="Save the processed frames as an annotated MP4.",
    )

    export_csv = st.toggle(
        "Enable CSV export",
        value=True,
    )


# ---------------------------------------------------------
# IMAGE MODE
# ---------------------------------------------------------

if mode == "Image":
    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if image_file is not None:
        original = Image.open(image_file).convert("RGB")

        frame = cv2.cvtColor(
            np.array(original),
            cv2.COLOR_RGB2BGR,
        )

        with st.spinner("Detecting handrail..."):
            result = model.predict(
                source=frame,
                conf=confidence,
                iou=iou,
                imgsz=image_size,
                verbose=False,
                retina_masks=True,
            )[0]

            annotated, detections = draw_segmentation_results(
                frame.copy(),
                result,
                show_masks=show_masks,
                show_boxes=show_rail_boxes,
                show_labels=show_rail_boxes,
                show_conf=show_rail_boxes,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original image")
            st.image(original, use_container_width=True)

        with col2:
            st.subheader("Handrail prediction")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        st.metric(
            "Handrail instances detected",
            len(detections),
        )

        success, encoded = cv2.imencode(".png", annotated)

        if success:
            st.download_button(
                "⬇ Download annotated image",
                data=encoded.tobytes(),
                file_name="annotated_handrail.png",
                mime="image/png",
            )

        if export_csv:
            st.download_button(
                "⬇ Download detection CSV",
                data=pd.DataFrame(detections).to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="handrail_image_detections.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------
# VIDEO MODE
# ---------------------------------------------------------

else:
    video_file = st.file_uploader(
        "Upload a CCTV video",
        type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"],
        key="video_upload",
    )

    if video_file is not None:
        st.video(video_file.getvalue())

        if st.button("▶ Run video analysis", type="primary"):
            workdir = Path(
                tempfile.mkdtemp(prefix="handrail_test_")
            )

            video_path = workdir / Path(video_file.name).name
            video_path.write_bytes(video_file.getvalue())

            info = get_video_info(str(video_path))

            if (
                info["frame_count"] <= 0
                or info["width"] <= 0
                or info["height"] <= 0
            ):
                st.error(
                    "Could not read this video. Try an H.264 MP4."
                )
                st.stop()

            pose_model = None

            if use_pose:
                with st.spinner("Loading pose model..."):
                    try:
                        pose_model = YOLO(pose_model_name)
                    except Exception as exc:
                        st.error(
                            f"Could not load pose model: {exc}"
                        )
                        st.stop()

            output_path = (
                workdir / "annotated_handrail_video.mp4"
            )

            writer = None

            if record_annotated_video:
                writer = open_video_writer(
                    str(output_path),
                    info["fps"],
                    info["width"],
                    info["height"],
                )

                if writer is None:
                    st.error(
                        "Could not initialize the video writer."
                    )
                    st.stop()

            cap = cv2.VideoCapture(str(video_path))

            progress = st.progress(0.0)
            status = st.empty()
            preview = st.empty()

            rail_tracker = PersistentRailMask(
                hold_frames=persist_frames
            )

            detections_log = []
            classification_log = []

            frame_index = 0
            handrail_total = 0
            using_total = 0
            violation_total = 0
            total_frames = info["frame_count"]

            try:
                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    frame_index += 1

                    # 1. Run handrail segmentation.
                    seg_result = model.predict(
                        source=frame,
                        conf=confidence,
                        iou=iou,
                        imgsz=image_size,
                        verbose=False,
                        retina_masks=True,
                    )[0]

                    # Count current detections regardless of display settings.
                    _, found = draw_segmentation_results(
                        frame.copy(),
                        seg_result,
                        show_masks=False,
                        show_boxes=False,
                        show_labels=False,
                        show_conf=False,
                    )

                    handrail_total += len(found)

                    for item in found:
                        detections_log.append({
                            "frame": frame_index,
                            "time_seconds": round(
                                (frame_index - 1) / info["fps"],
                                3,
                            ),
                            **item,
                        })

                    # 2. Update persistent rail mask.
                    rail_mask = rail_tracker.update(
                        seg_result,
                        frame.shape,
                    )

                    annotated = frame.copy()

                    if show_masks:
                        annotated = draw_persistent_rail(
                            annotated,
                            rail_mask,
                        )

                    if show_rail_boxes:
                        annotated, _ = draw_segmentation_results(
                            annotated,
                            seg_result,
                            show_masks=False,
                            show_boxes=True,
                            show_labels=True,
                            show_conf=True,
                        )

                    frame_using = 0
                    frame_violations = 0

                    # 3. Track people and classify wrist proximity.
                    if pose_model is not None:
                        pose_result = pose_model.track(
                            source=frame,
                            persist=True,
                            tracker=tracker,
                            conf=pose_confidence,
                            imgsz=image_size,
                            verbose=False,
                        )[0]

                        annotated, frame_using, frame_violations = (
                            classify_and_draw_people(
                                annotated,
                                pose_result,
                                rail_mask,
                                near_distance=near_distance,
                                wrist_confidence=wrist_confidence,
                                show_ids=show_track_ids,
                            )
                        )

                        using_total += frame_using
                        violation_total += frame_violations

                        classification_log.append({
                            "frame": frame_index,
                            "time_seconds": round(
                                (frame_index - 1) / info["fps"],
                                3,
                            ),
                            "people_using_handrail": frame_using,
                            "people_flagged_violation": frame_violations,
                        })

                    # 4. Record the annotated frame.
                    if writer is not None:
                        writer.write(annotated)

                    # 5. Update live preview.
                    preview_interval = max(
                        1, int(info["fps"])
                    )

                    if (
                        frame_index == 1
                        or frame_index % preview_interval == 0
                    ):
                        preview.image(
                            cv2.cvtColor(
                                annotated,
                                cv2.COLOR_BGR2RGB,
                            ),
                            caption=f"Processed frame {frame_index}",
                            use_container_width=True,
                        )

                    if frame_index % 5 == 0:
                        progress.progress(
                            min(
                                frame_index / max(total_frames, 1),
                                1.0,
                            )
                        )

                        status.caption(
                            f"Frames: {frame_index:,}/{total_frames:,} "
                            f"| Using: {frame_using} "
                            f"| Violations: {frame_violations}"
                        )

            finally:
                cap.release()

                if writer is not None:
                    writer.release()

            progress.progress(1.0)

            st.success(
                f"Finished processing {frame_index:,} frames."
            )

            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------

            st.subheader("Analysis summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Frames processed", f"{frame_index:,}")
            col2.metric("Handrail detections", f"{handrail_total:,}")
            col3.metric("Using classifications", f"{using_total:,}")
            col4.metric("Violation classifications", f"{violation_total:,}")

            st.caption(
                "Classification totals count person-frame results, "
                "not unique people or distinct violation events."
            )

            # -------------------------------------------------
            # ANNOTATED VIDEO AND DOWNLOAD
            # -------------------------------------------------

            if (
                record_annotated_video
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                st.subheader("🎬 Annotated recording")

                video_bytes = output_path.read_bytes()

                st.video(video_bytes)

                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=video_bytes,
                    file_name="annotated_handrail_video.mp4",
                    mime="video/mp4",
                    type="primary",
                )

            # -------------------------------------------------
            # CSV EXPORTS
            # -------------------------------------------------

            if export_csv and detections_log:
                rail_df = pd.DataFrame(detections_log)

                st.subheader("Handrail detections")
                st.dataframe(
                    rail_df.head(500),
                    use_container_width=True,
                )

                st.download_button(
                    "⬇ Download handrail CSV",
                    data=rail_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="handrail_detections.csv",
                    mime="text/csv",
                )

            if export_csv and classification_log:
                people_df = pd.DataFrame(classification_log)

                st.subheader("Person classification by frame")
                st.dataframe(
                    people_df.head(500),
                    use_container_width=True,
                )

                st.download_button(
                    "⬇ Download classification CSV",
                    data=people_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="handrail_classification.csv",
                    mime="text/csv",
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Green = USING HANDRAIL; red = VIOLATION based on wrist proximity. "
    "The classification is a heuristic and does not prove physical "
    "gripping. Persistent masks work best with stationary cameras."
)
