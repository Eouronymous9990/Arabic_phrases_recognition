import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from keras.models import load_model

dict = {
    0: "اريد التحدث",
    2: "السلام عليكم!",
    1: "عمل جيد!"
}

model = load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

font_path = r"C:\Users\zbook 17 g3\Desktop\cv\New folder (7)\Amiri-Bold.ttf"
font = ImageFont.truetype(font_path, 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    expresion = dict.get(class_id, "")
    class_name = expresion

    arabic_text = class_name
    reshaped_text = arabic_reshaper.reshape(arabic_text)
    bidi_text = get_display(reshaped_text)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    draw.text((50, 50), bidi_text, font=font, fill=(255, 255, 255))

    img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('Camera', img_with_text)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






