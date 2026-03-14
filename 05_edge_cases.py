import cv2
import os
from pathlib import Path

def extract_visual_proof(img_dir, custom_pred_dir, baseline_pred_dir, gt_dir, out_dir, conf_thresh=0.4):
    """
    Находит кадры для статьи, где ProxiBall нашел мяч, а базовая модель (Soccernet) промахнулась.
    Рисует рамки (Ground Truth, ProxiBall, Baseline) для наглядности.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
    
    found_cases = 0
    
    print("Начинаем поиск идеальных кадров для визуального анализа (Section 6.4)...")

    for img_path in img_paths:
        txt_name = img_path.stem + ".txt"
        
        custom_path = Path(custom_pred_dir) / txt_name
        baseline_path = Path(baseline_pred_dir) / txt_name
        gt_path = Path(gt_dir) / txt_name
        
        # Пропускаем картинки без мяча (Background)
        if not gt_path.exists() or os.path.getsize(gt_path) == 0:
            continue
            
        # Читаем предсказания (фильтруем по уверенности)
        custom_preds = []
        if custom_path.exists():
            with open(custom_path, 'r') as f:
                custom_preds = [line.strip().split() for line in f if float(line.split()[1]) >= conf_thresh]
                
        baseline_preds = []
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_preds = [line.strip().split() for line in f if float(line.split()[1]) >= conf_thresh]
        
        # ГЛАВНОЕ УСЛОВИЕ: ProxiBall нашел мяч, а Baseline полностью пропустил (False Negative)
        if len(custom_preds) > 0 and len(baseline_preds) == 0:
            found_cases += 1
            
            # Отрисовка для удобного визуального отбора
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]
            
            # Рисуем Ground Truth (синим)
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    x_c, y_c, w, h = map(float, parts[1:5])
                    x1, y1 = int((x_c - w/2) * img_w), int((y_c - h/2) * img_h)
                    x2, y2 = int((x_c + w/2) * img_w), int((y_c + h/2) * img_h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, "Ground Truth", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Рисуем предсказание ProxiBall (зеленым)
            for p in custom_preds:
                conf = float(p[1])
                x_c, y_c, w, h = map(float, p[2:6])
                x1, y1 = int((x_c - w/2) * img_w), int((y_c - h/2) * img_h)
                x2, y2 = int((x_c + w/2) * img_w), int((y_c + h/2) * img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"ProxiBall: {conf:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Сохраняем результат
            out_file = Path(out_dir) / img_path.name
            cv2.imwrite(str(out_file), img)

    print(f"\n[УСПЕХ] Найдено {found_cases} кадров, где ProxiBall сработал, а базовая модель нет.")
    print(f"Кадры сохранены в: {out_dir}")
    print("Просмотрите эту папку и выберите 3 лучших примера (Motion Blur, Truncated, Low Contrast) для статьи.")

if __name__ == "__main__":
    
    # ПУТИ ДЛЯ ВАШЕЙ СИСТЕМЫ
    img_dir = "/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/test/images"
    gt_dir = "/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/test/labels"
    
    # Сравниваем вашу лучшую модель с лучшим open-source конкурентом
    custom_preds = "/home/altay/Desktop/Footbonaut/6.1.data-eval/outputs/02_predictions/ProxiBall"
    baseline_preds = "/home/altay/Desktop/Footbonaut/6.1.data-eval/outputs/02_predictions/Soccernet"
    
    out_dir = "/home/altay/Desktop/Footbonaut/6.1.data-eval/outputs/05_visual_edge_cases"
    
    extract_visual_proof(img_dir, custom_preds, baseline_preds, gt_dir, out_dir)