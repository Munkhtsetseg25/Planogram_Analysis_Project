import cv2
import pandas as pd
from ultralytics import YOLO
import os

# ----------------- ТОХИРГОО -----------------

# ⚠️ 1. МОДЕЛИЙН ЗАМ (PATH): Raw string (r') ашиглан best.pt-н бүрэн замыг оруулсан.
MODEL_PATH = r'C:\Users\munkhtsetseg.b\Desktop\CM_Planogram\runs\detect\planogram_detection_final2\weights\best.pt' 

# ⚠️ 2. ШАЛГАХ ЗУРГИЙН ЗАМ: images/ фолдер доторх зургийг заана.
IMAGE_PATH = 'images/5d8006fb-64.jpg' 

# ⚠️ 3. АНГИЛЛЫН НЭРС (CLASS_NAMES): data.yaml файлтай яг таарч байх ёстой.
CLASS_NAMES = [
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
# --------------------------------------------

def analyze_shelf_occupancy(model_path, image_path, class_names):
    # 1. Моделийг ачаалах
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Модель ачаалахад алдаа гарлаа: {e}")
        return

    # 2. Зургийг ачаалж, нийт талбайг тооцоолох
    img = cv2.imread(image_path)
    if img is None:
        print(f"Зураг олдсонгүй эсвэл буруу зам: {image_path}")
        return

    # Зургийн нийт талбайг пикселээр тооцох
    H, W, _ = img.shape
    
    # Брэнд тус бүрийн эзэлсэн талбайг бүртгэх dictionary
    brand_area = {name: 0 for name in class_names}
    
    # 3. Объект илрүүлэх (Confidence-ийг 0.05 болгож бууруулсан)
    # 30 epoch-ийн моделд бага confidence-тэй ажиллах шаардлагатай.
    results = model(img, conf=0.05) 

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
                print(f"Анхааруулга: Class ID {class_id} нь ангиллын жагсаалтад байхгүй.")

    # 4. Тайлан гаргах
    report = []
    total_brand_area = sum(brand_area.values())
    
    if total_brand_area == 0:
        print("\nЗураг дээр ямар ч объект илрээгүй. (30 epoch-оор сураагүй байж магадгүй)")
        return

    # Хувийг тооцоолох
    for brand, area in brand_area.items():
        if area > 0:
            percentage = (area / total_brand_area) * 100
            report.append({
                'Брэнд': brand,
                'Эзэлсэн Талбай (Пиксель)': int(area),
                'Эзлэх Хувь (%)': round(percentage, 2)
            })

    # Үр дүнг Pandas DataFrame руу хөрвүүлэх
    df = pd.DataFrame(report)
    df = df.sort_values(by='Эзлэх Хувь (%)', ascending=False)
    
    print("\n--- ЛАНГУУНЫ АНАЛИЗЫН ТАЙЛАН ---")
    pd.set_option('display.max_rows', None) 
    print(df)
    
# Скриптийг ажиллуулах - Зөвхөн нэг удаа дуудагдах ёстой.
if __name__ == '__main__':
    analyze_shelf_occupancy(MODEL_PATH, IMAGE_PATH, CLASS_NAMES)