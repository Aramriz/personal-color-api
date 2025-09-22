import os
import io
import requests
import colorsys
from flask import Flask, request, jsonify
from google.cloud import vision
from PIL import Image
import concurrent.futures

app = Flask(__name__)
vision_client = vision.ImageAnnotatorClient()
print("✅ Google Vision client initialized successfully.")

# 🔧 Resize image
def resize_image(content, max_size=1024):
    image = Image.open(io.BytesIO(content))
    image.thumbnail((max_size, max_size))
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()

# 🎯 Crop region around facial landmark
def extract_face_region_color(content, landmark_type="LEFT_CHEEK_CENTER"):
    image = vision.Image(content=content)
    response = vision_client.face_detection(image=image)
    faces = response.face_annotations

    if not faces or not faces[0].landmarks:
        print("❌ ไม่พบใบหน้าหรือ landmark")
        return None

    landmark = next((lm for lm in faces[0].landmarks if lm.type_ == getattr(vision.FaceAnnotation.Landmark.Type, landmark_type)), None)
    if not landmark:
        print(f"❌ ไม่พบ landmark: {landmark_type}")
        return None

    img = Image.open(io.BytesIO(content))
    x, y = int(landmark.position.x), int(landmark.position.y)
    box_size = 20
    cropped = img.crop((x - box_size, y - box_size, x + box_size, y + box_size))

    output = io.BytesIO()
    cropped.save(output, format='JPEG')
    cropped_content = output.getvalue()

    cropped_image = vision.Image(content=cropped_content)
    color_response = vision_client.image_properties(image=cropped_image)
    return color_response.image_properties_annotation.dominant_colors.colors

# 🎨 Colormind API
def get_color_palette_from_colormind_async(base_colors):
    input_data = []
    for color in base_colors:
        rgb = color.color
        input_data.append([int(rgb.red), int(rgb.green), int(rgb.blue)])
    while len(input_data) < 5:
        input_data.append("N")

    def call_colormind():
        try:
            response = requests.post("http://colormind.io/api/", json={
                "model": "default",
                "input": input_data[:5]
            }, timeout=5)
            response.raise_for_status()
            return response.json().get("result")
        except Exception as e:
            print(f"❌ Colormind error: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(call_colormind)
        return future.result()

# 🔍 Undertone & Brightness
def classify_undertone(r, g, b):
    if r - b > 20 and r - g > 10:
        return "Warm"
    elif b - r > 20 and b - g > 10:
        return "Cool"
    else:
        return "Neutral"

def classify_brightness(l):
    if l > 0.65:
        return "Very Light"
    elif l > 0.55:
        return "Light"
    elif l > 0.4:
        return "Dark"
    else:
        return "Very Dark"

def analyze_personal_color(dominant_colors):
    undertone_counts = {"Warm": 0, "Cool": 0, "Neutral": 0}
    brightness_counts = {"Very Light": 0, "Light": 0, "Dark": 0, "Very Dark": 0}

    for color_info in dominant_colors:
        r, g, b = color_info.color.red, color_info.color.green, color_info.color.blue
        tone = classify_undertone(r, g, b)
        undertone_counts[tone] += 1
        h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
        brightness = classify_brightness(l)
        brightness_counts[brightness] += 1

    undertone = max(undertone_counts, key=undertone_counts.get)
    brightness = max(brightness_counts, key=brightness_counts.get)

    if undertone == "Warm":
        final_season = "Spring" if brightness in ["Very Light", "Light"] else "Autumn"
    else:
        final_season = "Summer" if brightness in ["Very Light", "Light"] else "Winter"

    return {"season": final_season, "undertone": undertone}

# 🌈 Seasonal color recommendations
color_recommendations = {
    "Autumn": ["Brown", "Caramel", "Mustard", "Olive Green", "Beige"],
    "Spring": ["Cream", "Light Orange", "Coral", "Peach", "Lime Green"],
    "Winter": ["Royal Blue", "True Red", "Lemon Yellow", "Hot Pink", "Pure White"],
    "Summer": ["Pastel Pink", "Sky Blue", "Seafoam", "Light Grey", "Lavender"]
}

# 📸 Main endpoint
@app.route("/analyze", methods=["POST"])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"}), 400

    image_file = request.files['image']
    original_content = image_file.read()
    content = resize_image(original_content)
    print("✅ 1. ภาพถูกย่อและรับเรียบร้อยแล้ว")

    try:
        print("⏳ 2. ตรวจสอบสีจากตำแหน่งบนใบหน้า...")
        landmark_type = request.form.get("landmark", "LEFT_CHEEK_CENTER")
        region_colors = extract_face_region_color(content, landmark_type)
        if not region_colors:
            return jsonify({"error": "ไม่พบสีจากตำแหน่งใบหน้า"}), 404
        print("✅ 2. ได้สีจากตำแหน่งเรียบร้อยแล้ว")

        print("⏳ 3. เรียก Colormind API...")
        final_palette = get_color_palette_from_colormind_async(region_colors)
        if not final_palette:
            return jsonify({"error": "ไม่สามารถสร้างพาเลตสีได้"}), 500
        print("✅ 3. ได้พาเลตสีเรียบร้อยแล้ว")

        analysis = analyze_personal_color(region_colors)
        season = analysis["season"]
        undertone = analysis["undertone"]
        color_names = color_recommendations.get(season, ["Color 1", "Color 2", "Color 3", "Color 4", "Color 5"])

        print(f"🎉 วิเคราะห์สำเร็จ: Season='{season}', Undertone='{undertone}'")
        return jsonify({
            "message": "วิเคราะห์สำเร็จ",
            "palette": final_palette,
            "undertone": undertone,
            "personalSeason": season,
            "colorNames": color_names
        })

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return jsonify({"error": f"เกิดข้อผิดพลาดระหว่างการวิเคราะห์: {e}"}), 500
    
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "🌈 Chariz Color API is alive and glowing!"
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)