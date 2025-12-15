import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os

# ----------------- ТОХИРГОО -----------------

# ⚠️ 1. МОДЕЛИЙН ЗАМ (PATH): Raw string (r') ашиглан best.pt-н бүрэн замыг оруулна.
# Таны 300 epoch-ийн сургалтын үр дүн: planogram_detection_final
MODEL_PATH = r'C:\Users\munkhtsetseg.b\Desktop\CM_Planogram\runs\detect\planogram_detection_final2\weights\best.pt' 

# ⚠️ 2. АНГИЛЛЫН НЭРС (CLASS_NAMES): data.yaml файлтай яг таарч байх ёстой.
CLASS_NAMES = [
    # Та нийт 159 нэрээ энд хуулж хийнэ!
    '100 naslaarai', '2080', 'A+', 'Pantene', 'ahmad', 'akbar', 'always', 'amber', 
    'anita', 'aquafresh', 'ariel', 'ariun', 'attack', 'babylab', 'belizna', 
    'bella', 'bibi', 'bimax', 'biomio', 'biomon', 'blend a med', 'blue touch', 
    'botanical garden', 'bro chips', 'bunny', 'carefree', 'classic', 'clean&white', 
    'clear', 'closeup', 'coffeeking', 'colgate', 'comet', 'comfort', 'cucu', 
    'daisy', 'discreet', 'divo', 'domestos', 'dove', 'duru', 'elis', 'elkos', 
    'elseve', 'enkhjin', 'fa', 'fairy', 'fasclean', 'first lady', 'flamingo', 
    'foxy', 'garnier', 'giggles', 'glade', 'gleace', 'goony', 'greenfield', 
    'gut&gunsting', 'hana', 'harmony', 'head&shoulders', 'huggies', 'ikh taiga', 
    'impra', 'java', 'jedentag', 'johnson''s', 'kerasys', 'khaantan', 'khatad', 
    'khuvsgul', 'kleenex bathroom paper', 'kleenex kt', 'kleenex tissue', 
    'kleenex wipes', 'kotex', 'lady', 'lanolovie', 'lays', 'liby', 'lipton', 
    'liq', 'living', 'loyd', 'lux', 'maccereal', 'mactea', 'mamypoko', 
    'maxkleen9', 'minime', 'moni happy', 'moony', 'mr.muscle', 'mungun ayga', 
    'naiman gishuun tuguldur', 'naturella', 'new top', 'nivea', 'nurse with ears', 
    'obuhiv', 'ok', 'ola', 'omo', 'oralb', 'pampers', 'panda', 'parodontax', 
    'pepsodent', 'persil', 'perwoll', 'popular', 'pringles', 'pronto', 'protex', 
    'ps', 'rascal friends', 'rascals', 'red', 'renova', 'romano', 'safeguard', 
    'sanitas', 'saraana', 'sarma', 'selpak', 'sensodyne', 'silk sense', 'sir', 
    'soffione', 'sofy body fit', 'sorti', 'splat', 'stimo', 'sunsilk', 'super', 
    'surf', 'syoss', 'tanay', 'tastea', 'tess', 'tide', 'tod', 'tody', 
    'toilet duck', 'toorkhon', 'tos', 'tresemme', 'trio', 'ud', 'umka', 
    'unibaby', 'unidry', 'urin', 'ushyasti nyn', 'vernel', 'viso', 'white', 
    'ya rodilsya', 'zero'
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
                continue
    
    total_brand_area = sum(brand_area.values())
    
    report = []
    if total_brand_area > 0:
        for brand, area in brand_area.items():
            if area > 0:
                percentage = (area / total_brand_area) * 100
                report.append({
                    # ⚠️ ЗАСВАР: Баганын нэрийг англиар болгов.
                    'Brand': brand,
                    'Occupancy (%)': round(percentage, 2)
                })
        df = pd.DataFrame(report)
        # ⚠️ ЗӨВХӨН ЭНД ҮЛДЭЭСЭН ХУВЬСАГЧИЙГ АШИГЛАН СОРТЛОНО.
        df = df.sort_values(by='Occupancy (%)', ascending=False).reset_index(drop=True) 
    else:
        # ⚠️ ЗАСВАР: Хоосон утгыг мөн англи баганын нэрээр үүсгэнэ.
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
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(plotted_image, caption='Брэнд Илрүүлэлтийн Үр Дүн', use_column_width=True)

        # -------------------------------------------------------------------------------------------------

        with col2:
            st.subheader("📊 Лангууны Эзлэх Хувийн Тайлан")
            
            # ⚠️ ЗАСВАР: Баганын нэр 'Occupancy (%)' болсон.
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
                # ⚠️ ЗАСВАР: x болон y тэнхлэгийн нэрийг англиар болгов.
                st.bar_chart(analysis_df, x='Brand', y='Occupancy (%)') 
            else:
                 st.warning("Зураг дээр ямар ч брэнд (объект) илрээгүй.")