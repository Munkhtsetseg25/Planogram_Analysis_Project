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

# ... (IMPORT, MODEL_PATH, CLASS_NAMES хэсгүүд хэвээр үлдэнэ) ...

# ----------------- STREAMLIT ИНТЕРФЭЙС (ШИНЭЧЛЭГДСЭН) -----------------

st.set_page_config(layout="wide")
st.title("🛒 Планограмын Автомат Анализ")
st.caption("Олон лангууны зургийг нэг дор оруулаад, нэгдсэн тайланг Excel-ээр татна уу.")

yolo_model = load_model()

# ... (Эхний хэсэг хэвээр үлдэнэ) ...

if yolo_model:
    uploaded_files = st.file_uploader(
        "Лангууны зургуудыг сонгох (.jpg, .png)", 
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=True 
    )

    if uploaded_files:
        all_results_df = []
        
        st.subheader("🖼️ Зураг Тус Бүрийн Анализ")
        
        # ⚠️ ЭНДЭЭС ГОЛ ӨӨРЧЛӨЛТ ЭХЭЛНЭ
        # Зураг бүрийг боловсруулж, хажууд нь тайланг нь харуулна.
        for uploaded_file in uploaded_files:
            st.markdown(f"---") # Зураг бүрийг ялгах зорилгоор зураас нэмэв
            st.markdown(f"**Зургийн Нэр:** `{uploaded_file.name}`")
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            plotted_image, analysis_df = process_image(img_array, yolo_model, CLASS_NAMES)
            
            analysis_df.insert(0, 'Filename', uploaded_file.name)
            all_results_df.append(analysis_df)

            # ⚠️ col1, col2-г loop дотор үүсгэж, зураг тус бүрийн хажууд тайланг харуулна.
            col_img, col_report = st.columns([2, 1]) # Зураг 2/3, Тайлан 1/3

            with col_img:
                st.image(plotted_image, caption=f'{uploaded_file.name} - Илрүүлэлтийн Үр Дүн', width=600)
                # width=600 нь зурагны хэмжээг жижигрүүлж, цагаан зайг багасгана.
            
            with col_report:
                st.markdown("##### 📊 Эзлэх Хувийн Тайлан")
                if analysis_df["Occupancy (%)"].sum() > 0:
                    st.dataframe(analysis_df[['Brand', 'Occupancy (%)']], use_container_width=True) # Зөвхөн брэнд, хувийг харуулна
                    st.bar_chart(analysis_df.head(10), x='Brand', y='Occupancy (%)', use_container_width=True)
                else:
                    st.warning(f"'{uploaded_file.name}' зураг дээр брэнд илрээгүй.")
        
        # ------------------ НЭГДСЭН ТАЙЛАН ҮҮСГЭХ ХЭСЭГ (ЭНЭ ХЭСЭГ ДООРОО ХЭВЭЭР ҮЛДЭНЭ) -------------------
        
        st.markdown("---")
        st.subheader("✅ Бүх Зургийн Нэгдсэн Анализын Үр Дүн")
        
        if all_results_df:
            final_df = pd.concat(all_results_df, ignore_index=True)
            
            st.markdown("### 1. Зураг Бүрийн Дэлгэрэнгүй Тайлан (Raw Data - Хүснэгт)")
            st.dataframe(final_df, use_container_width=True)

            summary_df = final_df.groupby('Brand').agg(
                Count=('Filename', 'count'), 
                Avg_Occupancy=('Occupancy (%)', 'mean')
            ).reset_index()
            summary_df = summary_df.sort_values(by='Avg_Occupancy', ascending=False).reset_index(drop=True)
            
            st.markdown("### 2. Брэнд Бүрийн Дундаж Эзлэх Хувь (Хүснэгт)")
            if summary_df["Avg_Occupancy"].sum() > 0:
                st.dataframe(summary_df, use_container_width=True)
                
                @st.cache_data
                def convert_df_to_excel(final_df): # Зөвхөн final_df-г авна
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        final_df.to_excel(writer, sheet_name='Raw_Data_Per_Image', index=False)
                    processed_data = output.getvalue()
                    return processed_data

                excel_data = convert_df_to_excel(final_df) # Зөвхөн final_df-г дамжуулна

                st.download_button(
                    label="📥 Дэлгэрэнгүй Тайланг Excel-ээр татах (Download)",
                    data=excel_data,
                    file_name='planogram_batch_raw_data.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
                
                st.subheader("3. Брэнд Бүрийн Дундаж Эзлэх Хувь (График)")
                top_10_summary = summary_df.head(10)
                st.bar_chart(top_10_summary, x='Brand', y='Avg_Occupancy', use_container_width=True) 
            else:
                st.warning("Оруулсан зургуудад ямар ч брэнд (объект) илрээгүй.")
        else:
            st.info("Анализ хийх зураг сонгоно уу.") # Хэрэв uploaded_files хоосон байвал