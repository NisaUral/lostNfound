"""
First baseline for the lost-item pilot: SigLIP-2 matching, no training.

Setup (once):   pip install torch transformers pillow numpy openpyxl
Run:            python baseline_eval.py --xlsx lost_item_pilot_dataset.xlsx --photos PATH\\TO\\categories
Optional:       --distractors PATH\\TO\\extra_images   (extra gallery images from the internet)
                --model google/siglip2-base-patch16-224   (default)
                --dummy   (no model; checks that the data and metrics code work)

What it measures (all item-level; the gallery has one F photo per item, so chance is 1/N):
  text -> photo : each Turkish description searches the F1 gallery and the F2 gallery
  photo -> photo: each L photo searches the F1 gallery and the F2 gallery
Metrics: Recall@1/3/5/10, MRR, Hard Pass Rate@5 (an item counts only if all its descriptions are in the top 5).
Breakdowns: category, description specificity, description type, same_background flag of the target photo.
All rows are saved to results.csv so you can look at the mistakes.
"""
import argparse, collections, csv, os, random, unicodedata
import numpy as np
from openpyxl import load_workbook
from PIL import Image

KS = (1, 3, 5, 10)


def read_sheet(wb, name):
    rows = list(wb[name].iter_rows(values_only=True))
    hdr = [str(h).split(" (")[0] if h else "" for h in rows[0]]
    return [dict(zip(hdr, r)) for r in rows[1:] if r[0] not in (None, "")]


def norm_name(s):
    s = s.replace("#U0130", "I")  # some unzip tools write the dotted capital I like this
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.replace("ACCEESSORIES", "ACCESSORIES").lower()


def resolve(folder, name):
    p = os.path.join(folder, name)
    if os.path.exists(p):
        return p
    target = norm_name(name)
    for f in os.listdir(folder):
        if norm_name(f) == target:
            return os.path.join(folder, f)
    raise FileNotFoundError(f"{name} not found in {folder}")


def tr_lower(s):
    return s.replace("İ", "i").replace("I", "ı").lower()


class SigLIP:
    def __init__(self, name, device=None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def _norm(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return x.cpu().numpy()

    def images(self, paths, item_ids=None, bs=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), bs):
                ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
                inp = self.proc(images=ims, return_tensors="pt").to(self.device)
                out.append(self._norm(self.model.get_image_features(**inp)))
        return np.concatenate(out)

    def texts(self, texts, item_ids=None, bs=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                inp = self.proc(text=[tr_lower(t) for t in texts[i:i + bs]], padding="max_length",
                                max_length=64, truncation=True, return_tensors="pt").to(self.device)
                out.append(self._norm(self.model.get_text_features(**inp)))
        return np.concatenate(out)


class Dummy:
    """Random vectors tied to item ids. Only for testing the code, the numbers mean nothing."""
    def __init__(self, dim=64, noise=1.0, seed=0):
        self.rng = np.random.default_rng(seed); self.base = {}; self.dim = dim; self.noise = noise

    def _vec(self, it):
        if it not in self.base:
            self.base[it] = self.rng.normal(size=self.dim)
        v = self.base[it] + self.noise * self.rng.normal(size=self.dim)
        return v / np.linalg.norm(v)

    def images(self, paths, item_ids=None, **kw):
        return np.stack([self._vec(i) for i in item_ids])

    def texts(self, texts, item_ids=None, **kw):
        return np.stack([self._vec(i) for i in item_ids])


def item_ranking(q, G, g_items):
    sims = G @ q
    best = {}
    for it, s in zip(g_items, sims):
        best[it] = max(best.get(it, -9), float(s))
    return sorted(best, key=lambda k: -best[k])


def summarize(rows, key, title):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r["rank"])
    print(f"\n{title}")
    print(f"{'':32}{'n':>5}" + "".join(f"{'R@'+str(k):>8}" for k in KS) + f"{'MRR':>8}")
    for g in sorted(groups, key=str):
        rk = np.array(groups[g])
        line = f"{str(g):32}{len(rk):>5}" + "".join(f"{(rk <= k).mean():>8.2f}" for k in KS)
        print(line + f"{(1 / rk).mean():>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--distractors", default=None)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    wb = load_workbook(a.xlsx, data_only=True)
    items = {r["item_id"]: r for r in read_sheet(wb, "Items")}
    photos = read_sheet(wb, "Photos")
    descs = [d for d in read_sheet(wb, "Descriptions") if d.get("text_tr")]
    print(f"{len(items)} items, {len(photos)} photos, {len(descs)} descriptions")

    emb = Dummy() if a.dummy else SigLIP(a.model)

    paths = {p["photo_file"]: resolve(a.photos, p["photo_file"]) for p in photos}
    pf = [p["photo_file"] for p in photos]
    P = emb.images([paths[f] for f in pf], [p["item_id"] for p in photos])
    pvec = dict(zip(pf, P))
    D = emb.texts([d["text_tr"] for d in descs], [d["item_id"] for d in descs])

    extra_vecs, extra_items = [], []
    if a.distractors:
        files = sorted(os.listdir(a.distractors))
        extra_vecs = emb.images([os.path.join(a.distractors, f) for f in files], [f"DIST_{i}" for i in range(len(files))])
        extra_items = [f"DIST_{i}" for i in range(len(files))]
        print(f"{len(files)} distractor images added to every gallery")

    rows = []
    for role in ("F1", "F2"):
        gal = [p for p in photos if p["role"] == role]
        G = np.stack([pvec[p["photo_file"]] for p in gal] + list(extra_vecs))
        g_items = [p["item_id"] for p in gal] + extra_items
        sb = {p["item_id"]: p.get("same_background") for p in gal}

        def add(qtype, qid, item, vec, d=None):
            order = item_ranking(vec, G, g_items)
            rows.append(dict(gallery=role, qtype=qtype, query=qid, item=item,
                             category=items[item]["category"], same_background=sb.get(item),
                             specificity=(d or {}).get("specificity", ""), dtype=(d or {}).get("type", ""),
                             rank=order.index(item) + 1, top1=order[0], top3=" ".join(order[:3])))

        for d, v in zip(descs, D):
            add("text", d["text_tr"], d["item_id"], v, d)
        for p in photos:
            if p["role"] == "L":
                add("photo", p["photo_file"], p["item_id"], pvec[p["photo_file"]])

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n_gal = len(items) + len(extra_items)
    print(f"\nGallery size per run: {n_gal} items -> chance R@1 = {1 / n_gal:.2f}")
    for qtype in ("text", "photo"):
        for role in ("F1", "F2"):
            sub = [r for r in rows if r["qtype"] == qtype and r["gallery"] == role]
            summarize([dict(r, all="all") for r in sub], "all", f"{qtype} -> {role} gallery")
    for qtype in ("text", "photo"):
        sub = [r for r in rows if r["qtype"] == qtype]
        summarize(sub, "category", f"{qtype}: by category (both galleries)")
        summarize(sub, "same_background", f"{qtype}: by same_background of the target photo")
    sub = [r for r in rows if r["qtype"] == "text"]
    summarize(sub, "specificity", "text: by specificity")
    summarize(sub, "dtype", "text: by description type")

    for role in ("F1", "F2"):
        per = collections.defaultdict(list)
        for r in rows:
            if r["qtype"] == "text" and r["gallery"] == role:
                per[r["item"]].append(r["rank"])
        hp = np.mean([all(x <= 5 for x in v) for v in per.values()])
        print(f"\nHard Pass Rate@5, text -> {role}: {hp:.2f}")
    print(f"\nAll rows saved to {a.out}. Open it and sort by rank to see the mistakes.")


if __name__ == "__main__":
    main()