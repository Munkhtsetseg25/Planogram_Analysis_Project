import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os

# ----------------- ТОХИРГОО -----------------

# ⚠️ 1. МОДЕЛИЙН ЗАМ (PATH) ШИНЭЧИЛСЭН (V2 загвар руу)
# Сургалтын үр дүн: runs/detect/planogram_detection_final_V2/weights/best.pt
MODEL_PATH = 'runs/detect/planogram_detection_final_V2/weights/best.pt' 

# ⚠️ 2. АНГИЛЛЫН НЭРС (CLASS_NAMES) ШИНЭЧИЛСЭН (177 нэр)
CLASS_NAMES = [
    '100 naslaarai', '2080', 'A+', 'Pantene', 'ahmad', 'aiwibi', 'akbar', 'alken',
    'always', 'amber', 'anita', 'aquafresh', 'ariel', 'ariun', 'attack', 'babylab',
    'belizna', 'bella', 'besto', 'bibi', 'bimax', 'biomio', 'biomon', 'blend a med',
    'blue touch', 'botanical garden', 'bro chips', 'bunny', 'c&s', 'carefree',
    'ciptadent', 'classic', 'clean&white', 'clear', 'closeup', 'coffeeking',
    'colgate', 'comet', 'comfort', 'cucu', 'daisy', 'delbee', 'depend', 'discreet',
    'divo', 'domestos', 'dove', 'duru', 'elis', 'elkos', 'elseve', 'enchanteur',
    'enkhjin', 'fa', 'fairy', 'fasclean', 'first lady', 'flamingo', 'foxy', 'garnier',
    'giggles', 'glade', 'gleace', 'goony', 'greenfield', 'gut&gunsting', 'hana',
    'harmony', 'head&shoulders', 'huggies', 'ikh taiga', 'impra', 'java', 'jedentag',
    'johnson\'s', 'kerasys', 'khaantan', 'khatad', 'khuvsgul', 'kleenex bathroom paper',
    'kleenex kt', 'kleenex tissue', 'kleenex wipes', 'kotex', 'lady', 'lanolovie',
    'lays', 'liby', 'lipton', 'liq', 'living', 'loyd', 'lux', 'maccereal', 'mactea',
    'mamypoko', 'maxkleen9', 'may', 'minime', 'moni happy', 'moony', 'mr.muscle',
    'mungun ayga', 'naiman gishuun tuguldur', 'natur', 'naturella', 'navch', 'new top',
    'nivea', 'nurse with ears', 'ob', 'obuhiv', 'oday', 'ok', 'ola', 'omo', 'oralb',
    'palmolive', 'pampers', 'panda', 'parodontax', 'pepsodent', 'persil', 'perwoll',
    'popular', 'pringles', 'pronto', 'protex', 'ps', 'rascal friends', 'rascals',
    'red', 'renova', 'romano', 'safeguard', 'sanitas', 'saraana', 'sarma', 'selpak',
    'sensodyne', 'silk sense', 'sir', 'soffione', 'sofy body fit', 'sorti', 'splat',
    'stimo', 'sunsilk', 'super', 'surf', 'syoss', 'tanay', 'tastea', 'tess', 'tide',
    'tod', 'tody', 'toilet duck', 'toorkhon', 'tos', 'tresemme', 'trio', 'ud', 'umka',
    'unibaby', 'unidry', 'urin', 'ushyasti nyn', 'vanish', 'veiro', 'vernel', 'viso',
    'white', 'ya rodilsya', 'yoursun', 'yrgui', 'zero'
]
# ------------------------------------------------------------------------------------------------------

@st.cache_resource 
def load_model():
    """Моделийг зөвхөн нэг удаа ачаална."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Модель ачаалахад алдаа гарлаа: {e}")
        return None

def process_image(img_array, model, class_names):
    """Зургийг боловсруулж, илрүүлэлтийг зурж, тайланг гаргана."""
    
    # Моделийн ачаалалт, conf=0.25 (confidence threshold)
    results = model(img_array, conf=0.25)
    
    # Илрүүлсэн бүтээгдэхүүнүүдийг зураг дээр буулгах
    plotted_img = results[0].plot() 
    plotted_img = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB) 
    
    # Талбайн анализ хийх
    brand_area = {name: 0 for name in class_names}
    
    for r in results:
        boxes = r.boxes.xywh 
        classes = r.boxes.cls
        
        for box, cls in zip(boxes, classes):
            w_pix = box[2].item()
            h_pix = box[3].item()
            box_area = w_pix * h_pix
            class_id = int(cls.item())
            
            try:
                brand_name = class_names[class_id]
                brand_area[brand_name] += box_area
            except IndexError:
                # 177-с дээш индекс гарвал алгасна
                continue
    
    total_brand_area = sum(brand_area.values())
    
    report = []
    if total_brand_area > 0:
        for brand, area in brand_area.items():
            if area > 0:
                percentage = (area / total_brand_area) * 100
                report.append({
                    'Brand': brand,
                    'Occupancy (%)': round(percentage, 2)
                })
        df = pd.DataFrame(report)
        df = df.sort_values(by='Occupancy (%)', ascending=False).reset_index(drop=True) 
    else:
        df = pd.DataFrame([{"Brand": "No Detections", "Occupancy (%)": 0}]) 

    return plotted_img, df

# ----------------- STREAMLIT ИНТЕРФЭЙС -----------------

st.set_page_config(layout="wide")
st.title("🛒 Планограмын Автомат Анализ (YOLOv8)")
st.caption("Лангууны зургийг чирч оруулаад, анализын тайланг шууд харна уу.")

yolo_model = load_model()

if yolo_model:
    # Зураг оруулах талбар
    uploaded_file = st.file_uploader("Лангууны зургийг сонгох (.jpg, .png)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Зургийг уншиж numpy array болгох
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.subheader("🖼️ Анализ Хийгдэж Буй Зураг")

        # Зургийг боловсруулж, үр дүнг авах
        plotted_image, analysis_df = process_image(img_array, yolo_model, CLASS_NAMES)

        # Үр дүнг зэрэгцүүлэн харуулах
        col1, col2 = st.columns([2, 2])

        with col1:
            st.image(plotted_image, caption='Брэнд Илрүүлэлтийн Үр Дүн', width=800) 
            # Эсвэл use_column_width=False хийгээд зөвхөн col1-ийн өргөнийг өөрчилж болно.

        # -------------------------------------------------------------------------------------------------

        with col2:
            st.subheader("📊 Лангууны Эзлэх Хувийн Тайлан")
            
            if analysis_df["Occupancy (%)"].sum() > 0:
                st.dataframe(analysis_df)
                
                # Тайланг татах товч
                csv = analysis_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Excel/CSV-ээр татах (Download)",
                    data=csv,
                    file_name='planogram_analysis_report.csv',
                    mime='text/csv',
                )
                
                st.subheader("График Дүрслэл")
                # X тэнхлэг дээр 177 ангилал нэгэн зэрэг харагдахгүй тул
                # Энд зөвхөн ТОП 10 брэндийг харуулахыг зөвлөж байна.
                top_10_df = analysis_df.head(10)
                
                st.bar_chart(top_10_df, x='Brand', y='Occupancy (%)') 
            else:
                st.warning("Зураг дээр ямар ч брэнд (объект) илрээгүй.")