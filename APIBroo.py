import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import base64

# โหลดโมเดล
model = load_model('models/trash_classifier_model.h5')  # ใช้เส้นทางจากโฟลเดอร์ที่เก็บไฟล์ใน GitHub
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

# API สำหรับทำนาย
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

if __name__ == '__main__':
    app.run(debug=True)
