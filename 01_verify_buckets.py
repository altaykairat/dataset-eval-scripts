import cv2
from pathlib import Path

def verify_thresholds(img_dir, label_dir, out_dir):
    # Draws size and velocity buckets on images for manual verification.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img_paths = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
    
    # Adjustable Thresholds
    SIZE_THRESH = {'small': 0.00025, 'medium': 0.002} 
    VEL_THRESH = {'slow': 1.05, 'medium': 1.3}
    
    # Statistics summary
    stats = {
        'size': {'Small': 0, 'Med': 0, 'Large': 0},
        'vel': {'Slow': 0, 'Med': 0, 'Fast': 0}
    }
    
    for img_path in img_paths:
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists(): continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        img_h, img_w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls, x_c, y_c, w, h = map(float, parts[:5])
                
                # Size classification
                area = w * h
                size_cls = "Small" if area < SIZE_THRESH['small'] else "Med" if area < SIZE_THRESH['medium'] else "Large"
                color = (255, 0, 0) if size_cls == "Small" else (0, 255, 255) if size_cls == "Med" else (0, 0, 255)
                stats['size'][size_cls] += 1
                    
                # Velocity classification (Aspect Ratio Proxy)
                ratio = w / h if h > 0 else 1.0
                vel_cls = "Slow" if ratio < VEL_THRESH['slow'] else "Med" if ratio < VEL_THRESH['medium'] else "Fast"
                stats['vel'][vel_cls] += 1

                # Draw
                x1, y1 = int((x_c - w/2) * img_w), int((y_c - h/2) * img_h)
                x2, y2 = int((x_c + w/2) * img_w), int((y_c + h/2) * img_h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"S:{size_cls} V:{vel_cls} (R:{ratio:.1f})"
                cv2.putText(img, label_text, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        cv2.imwrite(str(Path(out_dir) / img_path.name), img)

    print("\n--- Distribution Summary ---")
    print(f"Size Distribution: {stats['size']}")
    print(f"Velocity Distribution: {stats['vel']}")
    print(f"Total Labels: {sum(stats['size'].values())}")

if __name__ == "__main__":
    verify_thresholds(
        '/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/test/images', 
        '/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/test/labels', 
        '/home/altay/Desktop/Footbonaut/6.1.data-eval/outputs/01_verification'
        )