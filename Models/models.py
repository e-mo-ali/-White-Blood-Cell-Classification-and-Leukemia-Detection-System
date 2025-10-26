# import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from blood_model import BloodCellCNN
import os
import cv2
import numpy as np
# ---------------- تحميل النماذج ----------------
yolo_L = YOLO("D:\\Dataset Lukemia\\Project\\runs\\detect\\leukemia_yolo7\\weights\\best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
model = BloodCellCNN(num_classes)
model.load_state_dict(torch.load("weights/blood_cell_weights.pth", map_location=device))
model.to(device)
model.eval()
yolo_WBC = YOLO("E:\\team\\Projet2_JY\\runs\\detect\\yolov8n_final_cpu9\\weights\\best.pt")
WBC_model = load_model("E:\\team\\best_model.h5")
# أسماء الفئات
class_names = ['ALL','AML','CLL','CML','WBC']
output_dir_WBC = './Output_WBC'
output_dir_Lukemia = './output_Lukemia'





# ---------------- Function ----------------
def process_Lukemia(imges_path,name):
    output_dir_Lukemia_f = os.path.join(output_dir_Lukemia,name)
    os.makedirs(output_dir_Lukemia_f, exist_ok=True)
    # تحضير التحويلات
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    predictions = []
    
    num = 1
    for imges in os.listdir(imges_path):
        img_path = os.path.join(imges_path, imges)
        img1 = Image.open(img_path).convert('L')
        input_tensor = transform(img1).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        diagnosis = class_names[predicted_class]
        predictions.append(diagnosis)

        # print(f"✅ Predicted class for image {num}: {diagnosis}")

        # YOLO للكشف فقط إذا كانت الصورة مرضية
        if diagnosis != "WBC":
            yolo_result = yolo_L(img_path)

            # حفظ الصورة بعد YOLO
            save_path = os.path.join(output_dir_Lukemia_f, f"yolo_result_{num}_{diagnosis}.jpg")
            yolo_result[0].save(save_path)
            # print(f"💾 YOLO result saved: {save_path}")
        # else:
        #     print(f"ℹ️ Image {num} was normal (WBC).")

        num += 1

    # --------- حساب التوزيع العام ----------
    counts = {cls: predictions.count(cls) for cls in class_names}
    total = sum(counts.values()) if sum(counts.values()) > 0 else 1
    percentages = {cls: round((counts[cls] / total) * 100, 2) for cls in class_names}

    # إنشاء DataFrame
    summary_df = pd.DataFrame({
        "Leukemia Type": list(counts.keys()),
        "Count": list(counts.values()),
        "Percentage (%)": list(percentages.values())
    })

    # حفظ النتائج كملف Excel أو CSV
    summary_path = os.path.join(output_dir_Lukemia_f, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    # print(f"📊 Summary saved: {summary_path}")

    return summary_df








def process_WBC(input_images_dir,name):
    output_dir_WBC_f = os.path.join(output_dir_WBC,name) 
    os.makedirs(output_dir_WBC_f, exist_ok=True)
    # print(f"✅ Output directory ready: {output_dir_WBC_f}")

    # تحميل الموديلات
    yolo_model = yolo_WBC
    classifier_model = WBC_model
    classifier_input_size = (224, 224)

    # أسماء الخلايا
    class_names = ['Normal_Neutrophils', 'Normal_Lymphocytes', 'Normal_Eosinophils',
                   'Normal_Monocytes', 'Normal_Basophils']

    # ربط YOLO مع CNN
    yolo_to_cnn_mapping = {
        'neutrophil': 'Normal_Neutrophils',
        'lymphocyte': 'Normal_Lymphocytes',
        'eosinophil': 'Normal_Eosinophils',
        'monocyte': 'Normal_Monocytes',
        'basophil': 'Normal_Basophils'
    }
    yolo_valid_classes = list(yolo_to_cnn_mapping.keys())

    # عدادات ابتدائية
    differential_count = {
        'Normal_Neutrophils': 0,
        'Normal_Lymphocytes': int(0.40 * 100),
        'Normal_Eosinophils': int(0.03 * 100),
        'Normal_Monocytes': 0,
        'Normal_Basophils': int(0.005 * 100)
    }
    cnn_differential_count = differential_count.copy()

    # دالة تجهيز الصورة لـ CNN
    def preprocess_for_classifier(image, target_size):
        image = cv2.resize(image, target_size)
        image = img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    # ===================== المعالجة لكل صورة =====================
    for image_name in os.listdir(input_images_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(input_images_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            # print(f"⚠️ Error loading {image_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # كشف YOLO
        results = yolo_model.predict(image_path, conf=0.25, iou=0.6)

        # عدادات لكل صورة
        image_differential_count = {cls: 0 for cls in class_names}
        image_cnn_count = {cls: 0 for cls in class_names}

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            annotated_image = image_rgb.copy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                yolo_class_name = yolo_model.names[class_ids[i]]
                if yolo_class_name not in yolo_valid_classes:
                    continue

                yolo_mapped_class = yolo_to_cnn_mapping[yolo_class_name]
                if yolo_mapped_class in ['Normal_Neutrophils', 'Normal_Monocytes']:
                    image_differential_count[yolo_mapped_class] += 1

                cropped_cell = image_rgb[y1:y2, x1:x2]
                if cropped_cell.size == 0:
                    continue

                processed_cell = preprocess_for_classifier(cropped_cell, classifier_input_size)
                prediction = classifier_model.predict(processed_cell, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                model_class_names = ['Normal_Neutrophils', 'Normal_Lymphocytes',
                                     'Normal_Eosinophils', 'Normal_Monocytes',
                                     'Normal_Basophils', 'Blast', 'RBC']
                cnn_predicted_class = model_class_names[predicted_class_idx] \
                    if predicted_class_idx < len(model_class_names) else 'Unknown'

                if cnn_predicted_class in ['Normal_Neutrophils', 'Normal_Monocytes']:
                    image_cnn_count[cnn_predicted_class] += 1

                # رسم الصندوق على الصورة
                label = f"YOLO: {yolo_mapped_class}, CNN: {cnn_predicted_class}"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # حفظ الصورة المعنونة
            output_image_path = os.path.join(output_dir_WBC_f, f"annotated_{image_name}")
            cv2.imwrite(output_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # تحديث العد الكلي
        for cls in ['Normal_Neutrophils', 'Normal_Monocytes']:
            differential_count[cls] += image_differential_count[cls]
            cnn_differential_count[cls] += image_cnn_count[cls]

    # ===================== إنشاء DataFrames =====================
    overall_cnn_differential_df = pd.DataFrame({
        'Cell Type': class_names,
        'Count': [cnn_differential_count[cls] for cls in class_names],
        'Percentage (%)': [(count / sum(cnn_differential_count.values()) * 100)
                           if sum(cnn_differential_count.values()) > 0 else 0
                           for count in cnn_differential_count.values()]
    })

    # حفظ النتائج
    output_csv = os.path.join(output_dir_WBC_f, "overall_cnn_differential.csv")
    overall_cnn_differential_df.to_csv(output_csv, index=False)
    # print(f"📊 Saved CNN differential results to {output_csv}")

    return overall_cnn_differential_df
# ---------------- Test ----------------
# imges ="D:\\Dataset Lukemia\\Project\\final_dataset\\TestImages"
# WBC_images = "E:\\To\\test_images"
# report_df = process_Lukemia(imges)
# report_df = process_WBC(WBC_images)
# print("\n📊 Summary Report:")
# print(report_df)
