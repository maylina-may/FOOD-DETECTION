from ultralytics import YOLO

from flask import Flask, request, jsonify

import numpy as np

import os

import cv2

from flask_cors import CORS

import requests

import io

import base64



app = Flask(__name__)

CORS(app)



KEY = "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"

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"



model_path = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")

modelyolo = YOLO(model_path)



class_names = [

    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',

    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',

    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',

    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'

]



def preprocess_image(image_data):

    image = np.frombuffer(image_data, np.uint8)

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image



def draw_boxes(image, results):

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])

        label = class_names[cls]

        conf = float(box.conf[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

        text = f"{label} ({conf:.2f})"

        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return image



def query_llm(prompt, context=""):

    system_message = "Kamu adalah asisten gizi makanan Indonesia yang membantu pengguna memahami informasi gizi dari makanan serta memberikan saran pola makan yang sehat."

    

    if context:

        prompt = f"{context}\n\nPengguna: {prompt}"

    

    response = requests.post(

        "https://api.groq.com/openai/v1/chat/completions",

        headers={

            "Authorization": f"Bearer {KEY}",

            "Content-Type": "application/json",

        },

        json={

            "model": MODEL,

            "messages": [

                {"role": "system", "content": system_message},

                {"role": "user", "content": prompt}

            ],

            "temperature": 0.7,

        },

        timeout=30,

    )

    completion = response.json()

    return completion["choices"][0]["message"]["content"]



@app.route("/detect-gizi", methods=["POST"])

def detect_gizi():

    image_file = request.files.get("image")

    chat_history = request.form.get("chat_history", "")

    

    if not image_file:

        return jsonify({"error": "No image file provided"}), 400



    image_data = image_file.read()

    image = preprocess_image(image_data)



    results = modelyolo(image)[0]



    detected_objects = []

    makanan_list = []

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cls = int(box.cls[0])

        label = class_names[cls]

        conf = float(box.conf[0])

        makanan_list.append(label)

        detected_objects.append({

            "nama": label,

            "confidence": conf,

            "bbox": [x1, y1, x2, y2]

        })



    # Prompt makanan yang terdeteksi ke LLM untuk gizi dalam format chat

    makanan_str = ', '.join(list(set(makanan_list)))  # Unique

    

    if len(makanan_list) == 0:

        prompt = "Saya tidak bisa mendeteksi makanan dalam gambar ini. Mohon unggah gambar yang berisi makanan dengan jelas."

    else:

        prompt = f"Dari gambar yang diunggah, saya mendeteksi makanan berikut: {makanan_str}. Bisakah kamu memberikan informasi tentang kandungan gizi dari makanan tersebut? Berikan juga saran pola makan yang sehat terkait makanan ini."



    response_text = query_llm(prompt, chat_history)

    boxed_image = draw_boxes(image.copy(), results)

    _, img_encoded = cv2.imencode('.jpg', boxed_image)

    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')



    return jsonify({

        "objects": detected_objects,

        "image": "data:image/jpeg;base64," + img_base64,

        "response": response_text,

        "detected_foods": makanan_list

    })



@app.route("/chat", methods=["POST"])

def chat():

    data = request.json

    user_message = data.get("message", "")

    chat_history = data.get("chat_history", "")

    

    if not user_message:

        return jsonify({"error": "No message provided"}), 400

    

    response = query_llm(user_message, chat_history)

    

    return jsonify({

        "response": response

    })



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)

