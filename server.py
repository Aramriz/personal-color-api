import os
import requests
import colorsys
import cv2  # นำกลับมา
import dlib # นำกลับมา
import numpy as np # นำกลับมา
from flask import Flask, request, jsonify
from google.cloud import vision

# --- การตั้งค่า ---
app = Flask(__name__)
# ไม่ต้องระบุ path แล้ว เพราะเราตั้งค่าใน Environment Variable
vision_client = vision.ImageAnnotatorClient()

# --- โหลดโมเดลของ Dlib ---
print("⏳ Loading Dlib models...")
# ตรวจสอบให้แน่ใจว่าไฟล์ shape_predictor_68_face_landmarks.dat อยู่ในโฟลเดอร์เดียวกับ server.py
dlib_face_detector = dlib.get_frontal_face_detector()
dlib_landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("✅ Dlib models loaded successfully.")

# --- ฟังก์ชันที่นำกลับมา ---
def extract_facial_colors(image_bytes):
    """
    ฟังก์ชันสำหรับตรวจจับใบหน้าและดึงสีจากส่วนแก้มและหน้าผาก
    """
    try:
        # 1. แปลงข้อมูล bytes เป็น image array ที่ OpenCV ใช้งานได้
        image_np_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ Error: Could not decode image from bytes.")
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. ตรวจจับใบหน้า
        faces = dlib_face_detector(img_rgb)
        if not faces:
            print("⚠️ No faces detected by Dlib.")
            return []

        face = faces[0] # ใช้ใบหน้าแรกที่เจอ
        landmarks = dlib_landmark_predictor(img_rgb, face)
        
        # 3. กำหนดพื้นที่ที่สนใจ (แก้มซ้าย, แก้มขวา, หน้าผาก)
        cheek_left_pts = landmarks.parts()[2].x, landmarks.parts()[29].y
        cheek_right_pts = landmarks.parts()[14].x, landmarks.parts()[29].y
        forehead_pts = landmarks.parts()[27].x, (landmarks.parts()[21].y + landmarks.parts()[22].y) // 2
        
        # 4. ดึงสีจากพื้นที่เหล่านั้น
        colors = []
        if 0 <= forehead_pts[1] < img_rgb.shape[0] and 0 <= forehead_pts[0] < img_rgb.shape[1]:
            colors.append(img_rgb[forehead_pts[1], forehead_pts[0]])
        if 0 <= cheek_left_pts[1] < img_rgb.shape[0] and 0 <= cheek_left_pts[0] < img_rgb.shape[1]:
            colors.append(img_rgb[cheek_left_pts[1], cheek_left_pts[0]])
        if 0 <= cheek_right_pts[1] < img_rgb.shape[0] and 0 <= cheek_right_pts[0] < img_rgb.shape[1]:
            colors.append(img_rgb[cheek_right_pts[1], cheek_right_pts[0]])
        
        # แปลงเป็น format ที่ Google Vision ใช้เพื่อให้เข้ากับโค้ดส่วนอื่น
        dominant_colors_from_face = []
        for color in colors:
            color_obj = vision.ColorInfo(color=vision.Color(red=color[0], green=color[1], blue=color[2]), pixel_fraction=1.0/len(colors))
            dominant_colors_from_face.append(color_obj)

        return dominant_colors_from_face

    except Exception as e:
        print(f"Error in extract_facial_colors: {e}")
        return []


def get_color_palette_from_colormind(base_colors):
    # (ฟังก์ชันนี้เหมือนเดิมทุกประการ)
    input_data = []
    for color in base_colors:
        rgb = color.color
        input_data.append([int(rgb.red), int(rgb.green), int(rgb.blue)])
    while len(input_data) < 5:
        input_data.append("N")
    colormind_api_url = "http://colormind.io/api/"
    request_body = {"model": "default", "input": input_data[:5]}
    try:
        response = requests.post(colormind_api_url, json=request_body)
        response.raise_for_status()
        return response.json().get("result")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Colormind API: {e}")
        return None

def analyze_personal_color(dominant_colors):
    # (ฟังก์ชันนี้เหมือนเดิมทุกประการ)
    warm_colors_count = 0
    cool_colors_count = 0
    light_colors_count = 0
    dark_colors_count = 0
    for color_info in dominant_colors:
        r, g, b = color_info.color.red, color_info.color.green, color_info.color.blue
        if r > b:
            warm_colors_count += 1
        else:
            cool_colors_count += 1
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        if l > 0.55:
            light_colors_count += 1
        else:
            dark_colors_count += 1
    undertone = "Warm" if warm_colors_count > cool_colors_count else "Cool"
    feature_brightness = "Light" if light_colors_count > dark_colors_count else "Dark"
    print(f"🎨 Analysis: Undertone={undertone}, Features={feature_brightness}")
    final_season = "Unknown"
    if undertone == "Warm":
        final_season = "Autumn" if feature_brightness == "Dark" else "Spring"
    else: # Cool
        final_season = "Winter" if feature_brightness == "Dark" else "Summer"
    return {"season": final_season, "undertone": undertone}


@app.route("/analyze", methods=["POST"])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400
    image_file = request.files['image']
    content = image_file.read()
    print("✅ 1. Image file received successfully.")
    
    try:
        # --- ขั้นตอนที่ 1: ใช้ Dlib เพื่อดึงสีจากใบหน้า ---
        print("⏳ 2. Using Dlib to extract facial colors...")
        dominant_colors = extract_facial_colors(content)
        
        # --- ถ้า Dlib หาไม่เจอ ให้ใช้ Google Vision API แทน ---
        if not dominant_colors:
            print("⚠️ Dlib failed or found no face. Falling back to Google Vision API.")
            print("⏳ 2. Calling Google Vision API...")
            image = vision.Image(content=content) 
            response = vision_client.image_properties(image=image)
            dominant_colors = response.image_properties_annotation.dominant_colors.colors
            if not dominant_colors:
                return jsonify({"error": "No dominant colors found by Dlib or Google Vision"}), 404
        
        print("✅ 2. Dominant colors extracted successfully.")
        
        # --- ส่วนที่เหลือเหมือนเดิม ---
        print("⏳ 3. Calling Colormind API...")
        final_palette = get_color_palette_from_colormind(dominant_colors)
        if not final_palette:
            return jsonify({"error": "Failed to generate palette from Colormind"}), 500
        print("✅ 3. Colormind API responded successfully.")

        analysis = analyze_personal_color(dominant_colors)
        calculated_season = analysis["season"]
        calculated_undertone = analysis["undertone"]
        
        color_recommendations = {
            "Autumn": ["Brown", "Caramel", "Mustard", "Olive Green", "Beige"],
            "Spring": ["Cream", "Light Orange", "Coral", "Peach", "Lime Green"],
            "Winter": ["Royal Blue", "True Red", "Lemon Yellow", "Hot Pink", "Pure White"],
            "Summer": ["Pastel Pink", "Sky Blue", "Seafoam", "Light Grey", "Lavender"]
        }
        calculated_color_names = color_recommendations.get(calculated_season, ["Color 1", "Color 2", "Color 3", "Color 4", "Color 5"])

        print(f"🎉 Final Analysis: Season='{calculated_season}', Undertone='{calculated_undertone}'")
        return jsonify({
            "message": "Analysis successful",
            "palette": final_palette,
            "undertone": calculated_undertone,
            "personalSeason": calculated_season,
            "colorNames": calculated_color_names
        })

    except Exception as e:
        print(f"❌❌❌ AN ERROR OCCURRED DURING ANALYSIS: {e}")
        return jsonify({"error": f"An unexpected error occurred during analysis: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)