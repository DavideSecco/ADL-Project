import os

root = "data/imagenet/train"  # cartella con le sottocartelle per classe
list_file = "data/imagenet/meta/train.txt"  # percorso atteso dai config

os.makedirs(os.path.dirname(list_file), exist_ok=True)

with open(list_file, "w") as f:
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in sorted(os.listdir(class_dir)):
            if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                rel_path = os.path.join(class_name, img_name)
                f.write(rel_path + "\n")

print(f"Creato {list_file}")
