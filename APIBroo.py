from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Welcome to the Flask API!'})

# GET method
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask!'})

# POST method
@app.route('/add', methods=['POST'])
def add():
    try:
        # รับข้อมูล JSON จาก request
        data = request.get_json()

        # ตรวจสอบว่า a และ b มีค่า
        a = data.get('a')
        b = data.get('b')

        # ถ้ามีค่าไม่ครบ หรือ a, b ไม่มีค่าให้ตอบกลับข้อผิดพลาด
        if a is None or b is None:
            return jsonify({'error': "Missing 'a' or 'b' in request"}), 400

        # คำนวณผลลัพธ์
        result = a + b
        return jsonify({'result': result})

    except Exception as e:
        # หากเกิดข้อผิดพลาดในการประมวลผล ให้ส่งข้อความ error
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
