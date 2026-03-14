import os

labels_dir = "/home/altay/Desktop/Footbonaut/6.1.data-eval/testbench/testbench/test/labels"
count = 0
for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(labels_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        changed = False
        for line in lines:
            parts = line.split()
            if parts and parts[0] == '1':
                parts[0] = '0'
                new_lines.append(" ".join(parts) + "\n")
                changed = True
            else:
                new_lines.append(line)
        
        if changed:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            count += 1

print(f"Updated {count} label files.")
