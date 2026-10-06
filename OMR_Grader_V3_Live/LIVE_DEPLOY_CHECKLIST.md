# Live deployment checklist

If your existing GitHub repository already contains the older OMR app, replace/upload these V3 files:

- `app.py`
- `omr.py`
- `live_scanner.py` **(new)**
- `sheet_template.py`
- `requirements.txt`
- `.streamlit/config.toml`

Then in Streamlit Community Cloud:

1. Open the deployed app.
2. If it does not rebuild automatically, reboot/redeploy the app from the Streamlit dashboard.
3. Confirm the dependency install includes `streamlit-webrtc` and `av`.
4. Open the deployed HTTPS link on the phone.
5. Go to **Live grading**.
6. Tap **START** on the camera component.
7. Allow camera permission.
8. Test with a printed sheet and keep all four corner squares visible.

## Expected live behavior

- `Show page 1`
- Green outline appears when the page is found.
- `Hold steady 1/4 ... 4/4`
- The page is graded automatically.
- The same paper cannot be graded repeatedly while it remains in view.
- The app says to remove the paper.
- When the paper disappears, the scanner arms itself for the next page/student.
- For 51–100 questions, it asks for Page 2 before showing the final student score.
- If an answer is ambiguous, live scanning pauses and the teacher reviews only that question.

## WebRTC note

The deployed site must be HTTPS for browser camera access. Streamlit Community Cloud provides HTTPS. The app includes a STUN configuration, but some restrictive school or cellular networks can still require a TURN relay service.
