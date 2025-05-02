import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import base64
import requests

# ฟังก์ชันสำหรับดาวน์โหลดไฟล์จาก Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(f"{URL}&id={file_id}", stream=True)
    
    # ตรวจสอบสถานะการดาวน์โหลด
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded file to {destination}")
    else:
        print("Failed to download file")
        raise Exception("Download failed")

# โหลดโมเดลจากไฟล์ .h5 (ก่อนทำการทำนาย)
def load_trash_classifier_model():
    # ลิงก์ไฟล์ Google Drive (id ของไฟล์ .h5)
    file_id = '1tyqKB1ZUIOctQASkU-d8KVccrs9Wc_wj'  # เปลี่ยนเป็น ID ของไฟล์ .h5 ของคุณ
    model_path = 'trash_classifier_model.h5'
    
    # ดาวน์โหลดไฟล์จาก Google Drive
    download_file_from_google_drive(file_id, model_path)
    
    # โหลดโมเดลจากไฟล์ .h5
    model = load_model(model_path)
    return model

# โหลดโมเดล
model = load_trash_classifier_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# สร้างแอป Flask
app = Flask(__name__)

# ฟังก์ชันสำหรับการทำนาย
def predict_image(img):
    img = cv2.resize(img, (224, 224))  # ปรับขนาดภาพ
    img = img / 255.0  # ปรับขนาดค่า pixel
    img = np.expand_dims(img, axis=0)  # เพิ่มมิติ
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    confidence = np.max(prediction)
    return class_label, confidence

# ฟังก์ชันสำหรับการวาดกรอบบนภาพ
def draw_bounding_box(img):
    height, width, _ = img.shape
    # กำหนดขนาดกรอบที่ต้องการ
    box_size = 224
    x1 = width // 2 - box_size // 2
    y1 = height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # วาดกรอบ
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # วาดกรอบสีเขียว
    return img, (x1, y1, x2, y2)

# API สำหรับทำนายภาพที่ได้รับจากกล้อง
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลภาพจาก JSON
        data = request.get_json()
        image_data = data['image']
        
        # แปลงจาก base64 เป็นรูปภาพ
        img_data = base64.b64decode(image_data.split(',')[1])  # ตัด "data:image/jpeg;base64,"
        img = Image.open(BytesIO(img_data))
        img = np.array(img)

        # วาดกรอบบนภาพ
        img_with_box, (x1, y1, x2, y2) = draw_bounding_box(img)
        
        # ตัดภาพที่อยู่ในกรอบเพื่อนำไปทำนาย
        roi = img_with_box[y1:y2, x1:x2]
        
        # ทำนาย
        class_label, confidence = predict_image(roi)
        
        # ส่งผลการทำนายกลับ
        return jsonify({
            "prediction": class_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ฟังก์ชันสำหรับการจับภาพจากกล้อง
def capture_image_from_camera():
    cap = cv2.VideoCapture(0)  # ใช้กล้องเว็บแคม
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # แสดงผลภาพจากกล้อง
        cv2.imshow("Camera", frame)

        # ส่งภาพไปยัง API ทุก 1 วินาที
        if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อหยุด
            # แปลงภาพจาก OpenCV เป็น base64
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            # ส่งคำขอ POST ไปยัง API
            response = requests.post("http://127.0.0.1:5000/predict", json={"image": "data:image/jpeg;base64," + img_base64})
            result = response.json()

            # แสดงผลการทำนาย
            if "prediction" in result:
                print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # เริ่มต้นการจับภาพจากกล้อง
    capture_image_from_camera()
    
    # เริ่มแอป Flask
    # app.run(debug=True)  # ใช้แค่ในการรัน Flask API, ถ้าคุณต้องการรันในโหมด production ควรใช้ app.run(host='0.0.0.0')
