import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("./model/flower_model.h5")

class_names = ["dandelion", "daisy", "tulips", "sunflower", "roses"]

IMG_SIZE = 224

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    img = image.load_img(io.BytesIO(file.read()), target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    pred_label = class_names[pred_class]
    confidence = float(np.max(preds))

    return jsonify({"predicted_class": pred_label, "confidence": round(confidence, 4)})


if __name__ == "__main__":
    app.run(debug=True)
