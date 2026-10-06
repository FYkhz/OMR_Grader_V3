# OMR Grader V3 — Live Camera

A Streamlit + OpenCV OMR prototype that scans an answer key once and then grades student answer sheets continuously from a phone or laptop camera.

## What changed in V3

- Student papers no longer need to be uploaded one at a time.
- Uses `streamlit-webrtc` for a continuous live camera feed.
- Automatically detects all four corner markers.
- Waits for the sheet to remain stable before capturing.
- Grades one paper only once, then waits for the paper to leave the camera view.
- For 51–100 questions, prompts for Page 1 and then Page 2 and combines them.
- Pauses only when an answer is blank, multiple, or unclear and asks the teacher to review it.
- Stores a live session score table that can be exported as CSV.

## Install locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the address Streamlit prints, usually `http://localhost:8501`.

## Phone use

Deploy the app to an HTTPS host such as Streamlit Community Cloud. Open the deployed link on the phone, select **Live grading**, press **START**, and allow camera permission. The app requests the rear-facing camera when the browser supports it.

## Normal workflow

1. Choose 10–100 questions.
2. Download and print the generated answer sheet(s).
3. Fill and upload/photograph the teacher answer key.
4. Confirm any unclear key marks and save the key.
5. Open **Live grading** and press START.
6. Hold the student paper so all four black corner squares are visible.
7. Hold still until the app captures it automatically.
8. Remove the paper after the score appears.
9. Place the next paper under the camera.

For exams above 50 questions, the app asks for Page 1 and Page 2 before calculating the final score.

## Important deployment note

Live camera streaming uses WebRTC. HTTPS is required for remote browser camera access. A STUN server is configured in the app. Some school/cellular networks may still require a TURN relay service if WebRTC cannot establish a peer connection.

## Prototype limitations

- This is still an MVP and should be tested against real classroom lighting, printers, pens, pencils, phone cameras, and damaged sheets before high-stakes use.
- The current version assumes the teacher presents multipage exams in the requested order.
- Student names/rosters are intentionally not included yet.
