from flask import Flask, request, jsonify

app = Flask(__name__)

# GET method
@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello from Flask!'})

# POST method
@app.route('/add', methods=['POST'])
def add():
    data = request.get_json()
    result = data['a'] + data['b']
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
