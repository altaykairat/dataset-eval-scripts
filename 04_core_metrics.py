import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. МАТЕМАТИЧЕСКИЕ ФУНКЦИИ
# ==========================================

def calculate_iou(box1, box2):
    """Стандартный Intersection over Union (IoU)."""
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-6)

def calculate_nwd(box1, box2, img_w=1920, img_h=1080, C=12.8):
    """Normalized Wasserstein Distance (NWD) для микро-объектов."""
    cx1, cy1, w1, h1 = box1[0]*img_w, box1[1]*img_h, box1[2]*img_w, box1[3]*img_h
    cx2, cy2, w2, h2 = box2[0]*img_w, box2[1]*img_h, box2[2]*img_w, box2[3]*img_h
    
    center_dist2 = (cx1 - cx2)**2 + (cy1 - cy2)**2
    shape_dist2 = ((w1 - w2)**2 + (h1 - h2)**2) / 4.0
    
    return np.exp(-np.sqrt(center_dist2 + shape_dist2) / C)

def compute_ap(recall, precision):
    """Вычисляет Average Precision (AP) как площадь под PR-кривой (AUC)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

# ==========================================
# 2. ПРОЦЕССОР МЕТРИК
# ==========================================

def evaluate_and_plot(models, gt_dir, preds_root_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gt_paths = list(Path(gt_dir).glob('*.txt'))
    
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    results_list = []
    
    plt.figure(figsize=(10, 8))
    plt.title('Graph 1: Precision-Recall Curve (IoU=0.5)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c', '#34495e', '#e74c3c']

    print("Начинаем строгий расчет COCO-метрик...\n")

    for idx, model_name in enumerate(models):
        pred_dir = Path(preds_root_dir) / model_name
        if not pred_dir.exists():
            continue
            
        all_preds = []
        total_gt = 0
        
        for gt_path in gt_paths:
            pred_path = pred_dir / gt_path.name
            
            with open(gt_path, 'r') as f:
                gts = [list(map(float, line.strip().split())) for line in f]
            total_gt += len(gts)
            
            preds = []
            if pred_path.exists():
                with open(pred_path, 'r') as f:
                    preds = [list(map(float, line.strip().split())) for line in f]
                    
            # Сортируем предсказания по уверенности для текущего кадра
            preds.sort(key=lambda x: x[1], reverse=True)
            
            iou_matches_img = np.zeros((len(preds), len(iou_thresholds)))
            nwd_matches_img = np.zeros(len(preds))

            # Строгий матчинг для IoU по каждому порогу отдельно
            for t_idx, thresh in enumerate(iou_thresholds):
                gt_matched_iou = [False] * len(gts)
                for p_idx, p in enumerate(preds):
                    best_iou, best_gt = 0, -1
                    for gt_idx, gt in enumerate(gts):
                        if not gt_matched_iou[gt_idx]:
                            iou = calculate_iou(gt[1:5], p[2:6])
                            if iou > best_iou:
                                best_iou, best_gt = iou, gt_idx
                    if best_iou >= thresh:
                        gt_matched_iou[best_gt] = True
                        iou_matches_img[p_idx, t_idx] = 1

            # Строгий матчинг для NWD (порог 0.5)
            gt_matched_nwd = [False] * len(gts)
            for p_idx, p in enumerate(preds):
                best_nwd, best_gt = 0, -1
                for gt_idx, gt in enumerate(gts):
                    if not gt_matched_nwd[gt_idx]:
                        nwd = calculate_nwd(gt[1:5], p[2:6])
                        if nwd > best_nwd:
                            best_nwd, best_gt = nwd, gt_idx
                if best_nwd >= 0.5:
                    gt_matched_nwd[best_gt] = True
                    nwd_matches_img[p_idx] = 1
                    
            for p_idx, p in enumerate(preds):
                all_preds.append((p[1], iou_matches_img[p_idx], nwd_matches_img[p_idx]))

        if len(all_preds) == 0 or total_gt == 0:
            continue

        # Глобальная сортировка всех предсказаний датасета по уверенности
        all_preds.sort(key=lambda x: x[0], reverse=True)
        iou_matrix = np.array([x[1] for x in all_preds]) 
        nwd_array = np.array([x[2] for x in all_preds])
        
        # --- Вычисление площадей под кривыми ---
        ap_per_thresh = []
        for i in range(len(iou_thresholds)):
            tps = np.cumsum(iou_matrix[:, i])
            fps = np.cumsum(1 - iou_matrix[:, i])
            recalls = tps / total_gt
            precisions = tps / (tps + fps + 1e-16)
            ap_per_thresh.append(compute_ap(recalls, precisions))
            if i == 0: 
                r_50, p_50 = recalls, precisions
                
        map50 = ap_per_thresh[0]
        map50_95 = np.mean(ap_per_thresh)
        
        nwd_tps = np.cumsum(nwd_array)
        nwd_fps = np.cumsum(1 - nwd_array)
        nwd_recalls = nwd_tps / total_gt
        nwd_precisions = nwd_tps / (nwd_tps + nwd_fps + 1e-16)
        nwd_map = compute_ap(nwd_recalls, nwd_precisions)
        
        color = '#e74c3c' if model_name == 'ProxiBall' else colors[idx % len(colors)]
        linewidth = 3 if model_name == 'ProxiBall' else 1.5
        alpha = 1.0 if model_name == 'ProxiBall' else 0.6
        plt.plot(r_50, p_50, label=f'{model_name} (mAP: {map50:.3f})', color=color, linewidth=linewidth, alpha=alpha)
        
        print(f"[{model_name}] mAP50: {map50:.3f} | mAP50-95: {map50_95:.3f} | NWD-mAP: {nwd_map:.3f}")

        results_list.append({
            'Model': model_name,
            'mAP@50': round(map50, 4),
            'mAP@50-95': round(map50_95, 4),
            'NWD-mAP (0.5)': round(nwd_map, 4)
        })

    # --- STYLING ---
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=3)
    plt.tight_layout()
    pr_curve_path = Path(out_dir) / 'Graph_1_PR_Curve.png'
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()

    if results_list:
        csv_path = Path(out_dir) / 'Table_1_Core_Metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\n[УСПЕХ] Graph 1 и Table 1 успешно сгенерированы и сохранены!")

if __name__ == "__main__":
    labels = "/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/test/labels"
    preds_root = "/home/altay/Desktop/Footbonaut/6.1.data-eval/outputs/02_predictions"
    outputs = "/home/altay/Desktop/Footbonaut/6.1.data-eval/outputs/04_core_metrics"
    
    models_to_eval = [
        "Soccernet", "DFL", "Football-Ball-Detection",
        "ISSIA", "Ball-Detection", "Test-Project", "ProxiBall"
    ]
    
    evaluate_and_plot(models_to_eval, labels, preds_root, outputs)