@echo off
python -m pip install -r requirements.txt
ipconfig | findstr /i "IPv4"
echo.
echo On your phone, open http://YOUR-PC-IP:8501 while connected to the same Wi-Fi.
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
