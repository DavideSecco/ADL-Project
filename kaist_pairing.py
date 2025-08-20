from pathlib import Path
import argparse
import sys

def glob_many(root, patterns):
    out = []
    for pat in patterns:
        out.extend(root.glob(pat))
    return sorted(out)

def list_pairs_flat(set_dir):
    set_dir = Path(set_dir)
    vis_root = set_dir / "visible"
    lwir_root = set_dir / "lwir"

    if not vis_root.exists() or not lwir_root.exists():
        print(f"[ERRORE] Cartelle mancanti. visible: {vis_root.exists()}  lwir: {lwir_root.exists()}")
        print(f"[INFO] Path usato: {set_dir.resolve()}")
        sys.exit(1)

    vis_imgs = glob_many(vis_root, ["*.jpg","*.JPG","*.png","*.PNG"])
    lwir_imgs = glob_many(lwir_root, ["*.jpg","*.JPG","*.png","*.PNG"])

    if not vis_imgs:
        print(f"[ERRORE] Nessun file immagine trovato in: {vis_root}")
    if not lwir_imgs:
        print(f"[ERRORE] Nessun file immagine trovato in: {lwir_root}")

    print(f"[INFO] Trovati {len(vis_imgs)} file in visible, {len(lwir_imgs)} in lwir")

    lwir_dict = {p.stem: p for p in lwir_imgs}
    pairs = [(v, lwir_dict[v.stem]) for v in vis_imgs if v.stem in lwir_dict]
    only_vis = [v.name for v in vis_imgs if v.stem not in lwir_dict]
    only_lwir = [l.name for l in lwir_imgs if l.stem not in {v.stem for v in vis_imgs}]

    return pairs, only_vis, only_lwir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    pairs, only_vis, only_lwir = list_pairs_flat(args.dataset_path)
    print(f"[OK] Coppie trovate: {len(pairs)}")
    for v, l in pairs[:10]:
        print(v.name, "<->", l.name)
    if only_vis:
        print(f"[WARN] Solo in visible (prime 5): {only_vis[:5]}")
    if only_lwir:
        print(f"[WARN] Solo in lwir (prime 5): {only_lwir[:5]}")
