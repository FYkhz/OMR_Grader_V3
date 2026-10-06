# Deploy the live OMR grader for phone use

## Streamlit Community Cloud

1. Put this folder in a GitHub repository.
2. Go to https://share.streamlit.io.
3. Create an app and select the repository.
4. Select `app.py` as the entry point.
5. Deploy.
6. Open the resulting HTTPS `streamlit.app` URL on your phone.
7. Go to **Live grading**, press **START**, and allow camera access.

The browser should prefer the rear camera because the app requests `facingMode: environment`.

## If the camera does not connect

The hosted site must use HTTPS. Streamlit Community Cloud does. WebRTC also needs network traversal. The app includes a public STUN server, but restrictive Wi-Fi or mobile networks can require a TURN server. Test on the actual school/mobile networks you plan to support.
