from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route('/api/version', methods=['GET'])
def get_version():
    try:
        with open('data.txt', 'r') as file:
            # 读取文件内容并解析成JSON
            data = file.read()
            version_info = json.loads(data)
        return jsonify(version_info)
    except IOError:
        return jsonify({'error': 'File not found or cannot be read'})
    except ValueError:
        return jsonify({'error': 'Invalid JSON format in the file'})

if __name__ == '__main__':
    app.run(debug=True)