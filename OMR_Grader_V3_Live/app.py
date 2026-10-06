from __future__ import annotations

import hashlib
from datetime import datetime

import av
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from live_scanner import LiveGradeController
from omr import (
    CHOICES,
    OMRProcessingError,
    answers_from_detections,
    build_page_layout,
    crop_question,
    decode_image_bytes,
    detection_rows,
    draw_overlay,
    encode_png,
    page_count_for,
    scan_sheet,
)
from sheet_template import create_sheet


st.set_page_config(page_title="OMR Live Grader", page_icon="📷", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1rem; padding-bottom: 3rem;}
.omr-hero {padding: 1rem 1.2rem; border: 1px solid #dbe5f4; border-radius: 18px; background: linear-gradient(135deg,#f7fbff,#eef5ff); margin-bottom: 1rem;}
.omr-hero h1 {margin: 0; font-size: 2.05rem; color: #0b3975;}
.omr-hero p {margin: .35rem 0 0; color: #40536d;}
[data-testid="stMetric"] {border: 1px solid #dfe6ef; padding: .65rem; border-radius: 12px; background: white;}
.live-ready {padding: .8rem 1rem; border-radius: 12px; border: 1px solid #cbe5d2; background:#f3fbf5;}
@media (max-width: 700px) {
  .block-container {padding-left: .65rem; padding-right: .65rem; padding-top: .55rem;}
  .omr-hero h1 {font-size: 1.55rem;}
  .omr-hero p {font-size: .90rem;}
  button {min-height: 46px;}
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="omr-hero">
  <h1>OMR Live Grader</h1>
  <p>Scan the answer key once, then grade student sheets continuously with the phone camera.</p>
</div>
""",
    unsafe_allow_html=True,
)


def initial_state() -> dict:
    return {
        "exam_questions": 20,
        "key_scans": {},
        "key_hashes": {},
        "key_overrides": {},
        "key_answers": None,
        "key_confirmed_at": None,
    }


def initialize_state() -> None:
    for key, value in initial_state().items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_exam(new_total: int) -> None:
    st.session_state.exam_questions = int(new_total)
    st.session_state.key_scans = {}
    st.session_state.key_hashes = {}
    st.session_state.key_overrides = {}
    st.session_state.key_answers = None
    st.session_state.key_confirmed_at = None
    st.session_state.pop("live_controller", None)
    st.session_state.pop("live_controller_signature", None)
    for key in list(st.session_state.keys()):
        if key.startswith(("key_p", "review_key_", "edit_key_", "live_review_")):
            st.session_state.pop(key, None)


initialize_state()


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def file_bytes(uploaded) -> bytes:
    uploaded.seek(0)
    return uploaded.read()


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_picker(prefix: str, page_number: int):
    upload_tab, camera_tab = st.tabs(["Upload answer-key image", "Take answer-key photo"])
    with upload_tab:
        upload = st.file_uploader(
            f"Upload answer-key page {page_number}",
            type=["jpg", "jpeg", "png"],
            key=f"{prefix}_upload",
        )
    with camera_tab:
        camera = st.camera_input(
            f"Photograph answer-key page {page_number}",
            key=f"{prefix}_camera",
            help="Keep the complete page and all four black corner squares visible.",
        )
    return camera if camera is not None else upload


def review_problem_answers(scan, *, group: str, allow_blank: bool) -> dict[int, str | None]:
    overrides: dict[int, str | None] = dict(st.session_state[group])
    problems = [d for d in scan.detections if d.status != "clear"]
    if not problems:
        st.success(f"All questions on page {scan.layout.page_number} were detected clearly.")
        return overrides

    st.warning(f"{len(problems)} answer-key question(s) need review.")
    allowed = list(CHOICES) + (["Blank"] if allow_blank else [])
    options = ["Select..."] + allowed

    for detection in problems:
        q = detection.question
        with st.expander(
            f"Question {q}: {detection.status.title()} · confidence separation {detection.confidence:.0f}%",
            expanded=True,
        ):
            st.image(bgr_to_rgb(crop_question(scan, q)), use_container_width=True)
            st.caption(
                "Ink density — "
                + " · ".join(
                    f"{choice}: {density * 100:.1f}%"
                    for choice, density in zip(CHOICES, detection.densities)
                )
            )
            current = overrides.get(q)
            index = options.index(current) if current in allowed else 0
            selected = st.radio(
                f"Confirmed answer for Question {q}",
                options,
                index=index,
                horizontal=True,
                key=f"review_{group}_{q}",
            )
            if selected == "Select...":
                overrides.pop(q, None)
            else:
                overrides[q] = selected

    st.session_state[group] = overrides
    return overrides


def optional_clear_edits(scan, *, group: str) -> dict[int, str | None]:
    overrides: dict[int, str | None] = dict(st.session_state[group])
    with st.expander(f"Optional: correct a clearly detected answer on page {scan.layout.page_number}"):
        st.caption("Only change a value if the image shows the automatic choice is wrong.")
        columns = st.columns(4)
        for index, detection in enumerate(scan.detections):
            if detection.status != "clear":
                continue
            q = detection.question
            current = overrides.get(q, detection.selected)
            selected = columns[index % 4].selectbox(
                f"Question {q}",
                list(CHOICES),
                index=list(CHOICES).index(current) if current in CHOICES else 0,
                key=f"edit_{group}_{q}",
            )
            if selected == detection.selected:
                overrides.pop(q, None)
            else:
                overrides[q] = selected
    st.session_state[group] = overrides
    return overrides


def scan_key_page_widget(page_number: int):
    total = st.session_state.exam_questions
    layout = build_page_layout(total, page_number)
    st.markdown(
        f"### Page {page_number} of {layout.page_count} — Questions {layout.first_question}–{layout.last_question}"
    )
    source = source_picker(f"key_p{page_number}", page_number)
    if source is not None:
        data = file_bytes(source)
        digest = image_hash(data)
        if st.session_state.key_hashes.get(page_number) != digest:
            try:
                with st.spinner(f"Reading answer-key page {page_number}..."):
                    image = decode_image_bytes(data)
                    result = scan_sheet(image, layout=layout, answer_key_mode=True)
                    st.session_state.key_scans[page_number] = result
                    st.session_state.key_hashes[page_number] = digest
                    q_range = range(layout.first_question, layout.last_question + 1)
                    st.session_state.key_overrides = {
                        q: a for q, a in st.session_state.key_overrides.items() if q not in q_range
                    }
                    st.session_state.key_answers = None
                    st.session_state.key_confirmed_at = None
                    st.session_state.pop("live_controller", None)
                    st.session_state.pop("live_controller_signature", None)
            except OMRProcessingError as exc:
                st.error(str(exc))

    scan = st.session_state.key_scans.get(page_number)
    if scan is None:
        st.info(f"Upload or photograph answer-key page {page_number}.")
        return None

    if scan.warning:
        st.warning(scan.warning)

    overlay = draw_overlay(scan, overrides=st.session_state.key_overrides)
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.image(bgr_to_rgb(overlay), caption=f"Detected key page {page_number}", use_container_width=True)
    with right:
        rows = pd.DataFrame(detection_rows(scan.detections))
        clear_count = int((rows["Status"] == "Clear").sum())
        m1, m2 = st.columns(2)
        m1.metric("Clear", clear_count)
        m2.metric("Review", len(rows) - clear_count)
        st.dataframe(
            rows[["Question", "Detected", "Status", "Confidence"]],
            hide_index=True,
            use_container_width=True,
            height=300,
        )

    review_problem_answers(scan, group="key_overrides", allow_blank=False)
    optional_clear_edits(scan, group="key_overrides")
    return scan


def get_live_controller() -> LiveGradeController:
    signature = (
        int(st.session_state.exam_questions),
        tuple(st.session_state.key_answers or []),
    )
    if (
        "live_controller" not in st.session_state
        or st.session_state.get("live_controller_signature") != signature
    ):
        st.session_state.live_controller = LiveGradeController(
            st.session_state.exam_questions,
            st.session_state.key_answers,
        )
        st.session_state.live_controller_signature = signature
    return st.session_state.live_controller


with st.sidebar:
    st.header("Exam")
    st.metric("Questions", st.session_state.exam_questions)
    st.caption(f"Pages required: {page_count_for(st.session_state.exam_questions)}")
    if st.session_state.key_answers:
        st.success("Answer key ready")
        st.caption(st.session_state.key_confirmed_at)
    else:
        st.info("Answer key not saved")
    st.divider()
    page = st.radio(
        "Workflow",
        ["1. Setup & sheets", "2. Answer key", "3. Live grading", "4. Results"],
        label_visibility="collapsed",
    )


if page == "1. Setup & sheets":
    st.subheader("Choose the exam size")
    chosen = st.number_input(
        "Total number of questions",
        min_value=10,
        max_value=100,
        value=int(st.session_state.exam_questions),
        step=1,
        help="Up to 50 questions are placed on each page. Exams over 50 use two pages.",
    )
    pages = page_count_for(int(chosen))
    st.info(f"{int(chosen)} questions will use {pages} answer-sheet page{'s' if pages > 1 else ''}.")

    if int(chosen) != st.session_state.exam_questions:
        st.warning("Changing the question count clears the saved answer key and live session.")
        if st.button("Apply question count", type="primary", use_container_width=True):
            reset_exam(int(chosen))
            st.rerun()

    st.divider()
    st.subheader("Download the printable answer sheets")
    st.write("Use the same sheet format for the teacher's answer key and each student paper.")
    for page_number in range(1, page_count_for(st.session_state.exam_questions) + 1):
        layout = build_page_layout(st.session_state.exam_questions, page_number)
        sheet = create_sheet(st.session_state.exam_questions, page_number)
        c1, c2 = st.columns([1.2, 0.8], gap="large")
        with c1:
            st.image(
                bgr_to_rgb(sheet),
                caption=f"Page {page_number}: Questions {layout.first_question}–{layout.last_question}",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                f"Download page {page_number} (PNG)",
                data=encode_png(sheet),
                file_name=f"omr_{st.session_state.exam_questions}q_page_{page_number}.png",
                mime="image/png",
                use_container_width=True,
            )
            st.markdown(
                "**Printing / scanning tips**\n\n"
                "- Keep all four black corner squares visible.\n"
                "- Do not crop or stretch the sheet when printing.\n"
                "- Fill bubbles darkly with pen or pencil.\n"
                "- Avoid strong shadows when using the phone camera."
            )

elif page == "2. Answer key":
    st.subheader("Scan and validate the answer key")
    st.write(
        "Fill one set of sheets with the correct answers. Upload or photograph each page, then confirm every unclear question."
    )
    pages = page_count_for(st.session_state.exam_questions)
    for page_number in range(1, pages + 1):
        with st.container(border=True):
            scan_key_page_widget(page_number)

    missing_pages = [p for p in range(1, pages + 1) if p not in st.session_state.key_scans]
    if missing_pages:
        st.error("Missing answer-key page(s): " + ", ".join(map(str, missing_pages)))
    else:
        all_answers: dict[int, str | None] = {}
        unresolved: list[int] = []
        for page_number in range(1, pages + 1):
            scan = st.session_state.key_scans[page_number]
            page_answers = answers_from_detections(scan.detections, st.session_state.key_overrides)
            all_answers.update(page_answers)
            unresolved.extend(
                d.question
                for d in scan.detections
                if d.status != "clear" and d.question not in st.session_state.key_overrides
            )

        missing_answers = [q for q in range(1, st.session_state.exam_questions + 1) if all_answers.get(q) is None]
        unresolved = sorted(set(unresolved + missing_answers))
        if unresolved:
            st.error("The key cannot be saved until these questions are confirmed: " + ", ".join(map(str, unresolved)))
        else:
            ordered = [str(all_answers[q]) for q in range(1, st.session_state.exam_questions + 1)]
            st.success("The answer key is complete and valid.")
            with st.expander("Review detected answer key"):
                st.code(" · ".join(f"{q}:{answer}" for q, answer in enumerate(ordered, start=1)), language=None)
            if st.button("Confirm and save answer key", type="primary", use_container_width=True):
                st.session_state.key_answers = ordered
                st.session_state.key_confirmed_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
                st.session_state.pop("live_controller", None)
                st.session_state.pop("live_controller_signature", None)
                st.success("Answer key saved. Live grading is ready.")
                st.rerun()

elif page == "3. Live grading":
    st.subheader("Live camera grading")
    if not st.session_state.key_answers:
        st.warning("Scan and confirm the answer key first.")
        st.stop()

    controller = get_live_controller()
    st.markdown(
        '<div class="live-ready"><b>How to use:</b> Press START, allow camera access, point the back camera at the complete sheet, and hold still. The app captures automatically. Remove the paper after the score appears.</div>',
        unsafe_allow_html=True,
    )
    if page_count_for(st.session_state.exam_questions) == 2:
        st.info("For this exam, scan Page 1 first. Remove it, then the app will ask for Page 2 and combine both pages into one score.")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        processed = controller.process_frame(image)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

    webrtc_streamer(
        key="omr-live-camera",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"facingMode": {"ideal": "environment"}},
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_toggle_controls=False,
    )

    if st.button("Reset current student scan", use_container_width=True):
        controller.reset_current_student()
        st.toast("Current scan reset.")

    @st.fragment(run_every=0.5)
    def live_status_panel():
        snap = controller.snapshot()
        st.divider()

        if snap.phase == "WAITING":
            st.info(f"📷 {snap.message}")
        elif snap.phase == "WAIT_REMOVE":
            st.success(f"✅ {snap.message}")
        elif snap.phase == "REVIEW":
            st.warning(f"⚠️ {snap.message}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Student scan", snap.student_number)
        c2.metric("Expected page", f"{snap.expected_page}/{snap.page_count}")
        c3.metric("Captured pages", len(snap.captured_pages))

        if snap.latest_grade:
            grade = snap.latest_grade
            st.markdown("### Last completed paper")
            a, b, c, d = st.columns(4)
            a.metric("Score", f"{grade['score']}/{grade['total']}")
            b.metric("Percentage", f"{grade['percentage']:.1f}%")
            c.metric("Incorrect", grade["incorrect"])
            d.metric("Blank", grade["blank"])

        if snap.phase == "REVIEW":
            st.markdown("### Questions needing teacher review")
            st.caption("The live scanner is paused until every orange/unclear question is confirmed.")
            selections: dict[int, str | None] = {}
            complete = True
            for item in controller.review_items():
                q = item["question"]
                with st.expander(
                    f"Question {q} · {item['status'].title()} · separation {item['confidence']:.0f}%",
                    expanded=True,
                ):
                    st.image(bgr_to_rgb(item["crop"]), use_container_width=True)
                    st.caption(
                        "Ink density — "
                        + " · ".join(
                            f"{choice}: {density * 100:.1f}%"
                            for choice, density in zip(CHOICES, item["densities"])
                        )
                    )
                    options = ["Select...", "A", "B", "C", "D", "Blank"]
                    selected = st.radio(
                        f"Final answer for Question {q}",
                        options,
                        horizontal=True,
                        key=f"live_review_{snap.student_number}_{q}",
                    )
                    if selected == "Select...":
                        complete = False
                    else:
                        selections[q] = selected

            if st.button(
                "Confirm reviewed answers and finish grading",
                type="primary",
                use_container_width=True,
                disabled=not complete,
                key=f"confirm_live_review_{snap.student_number}",
            ):
                controller.resolve_review(selections)
                st.rerun(scope="fragment")

        history = controller.history()
        if history:
            st.markdown("### Recent scans")
            st.dataframe(pd.DataFrame(history[-8:]), hide_index=True, use_container_width=True)

    live_status_panel()

elif page == "4. Results":
    st.subheader("Live session results")
    controller = st.session_state.get("live_controller")
    history = controller.history() if controller is not None else []
    if not history:
        st.info("No completed live scans are in this session yet.")
    else:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download session results (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="omr_live_session_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    f"Live OMR prototype · 10–100 questions · Up to 50 questions/page · Current exam: {st.session_state.exam_questions} questions"
)
