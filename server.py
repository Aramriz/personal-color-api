import os
import requests
import colorsys
from flask import Flask, request, jsonify
from google.cloud import vision

# --- การตั้งค่า ---
app = Flask(__name__)
# ไม่ต้องระบุ path แล้ว เพราะเราตั้งค่าใน Environment Variable บน Render
vision_client = vision.ImageAnnotatorClient()
print("✅ Google Vision client initialized successfully.")


# --- ฟังก์ชันที่ปรับปรุงแล้ว ---

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
        # --- ♻️ [ปรับปรุง] เรียกใช้ Google Vision API โดยตรง ---
        print("⏳ 2. Calling Google Vision API...")
        image = vision.Image(content=content) 
        response = vision_client.image_properties(image=image)
        dominant_colors = response.image_properties_annotation.dominant_colors.colors
        
        if not dominant_colors:
            return jsonify({"error": "No dominant colors found by Google Vision"}), 404
        
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)