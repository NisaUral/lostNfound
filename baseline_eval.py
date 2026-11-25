"""
Baseline for the lost-item pilot: SigLIP-2 matching, no training.

Setup (once):   pip install torch transformers pillow numpy openpyxl
Run:            python baseline_eval.py --xlsx DOSYA.xlsx --photos FOTOGRAF_KLASORU
Optional:       --hub-norm        also score with hub normalization (CSLS) and compare with plain scores
                --crop 4          also index up to 4 object crops per found photo (F1/F2) found by an open-vocabulary
                                  detector (OWLv2) and compare with the full-image gallery ("+crop" rows)
                --save-crops DIR  save the crops so you can look at them
                --distractors DIR extra gallery images added to every gallery
                --model google/siglip2-base-patch16-224     --detector google/owlv2-base-patch16-ensemble
                --keep-punct      do not strip commas / circumflexes before embedding
                --dummy           no models; only checks that the data and metrics code work

What it measures (item level; chance = 1/N for N gallery items):
  text -> photo : each description searches the F1 gallery and the F2 gallery
                  (if the sheet has a text_tr_clean column, both text versions are run)
  photo -> photo: each L photo searches the F1 gallery and the F2 gallery
Metrics: Recall@1/3/5/10, MRR, Hard Pass Rate@1/3/5, wrong top-1 winners (hub check), hardest target photos.
Everything is saved to results.csv.

Hub normalization (CSLS): score(q, g) = 2*cos(q, g) - r_g, r_g = mean of the top-k cosines between g and a bank of other
queries (never the query itself or the gallery photo's own item).
Crops ("+crop"): every found photo is indexed as the full image PLUS its object crops; an item's score is the best
score among them, so an object that is small or sits in a cluttered scene can still be found.
"""
import argparse, collections, csv, os, re, unicodedata
import numpy as np
from openpyxl import load_workbook
from PIL import Image


KS = (1, 3, 5, 10)


def read_sheet(wb, name):
    rows = list(wb[name].iter_rows(values_only=True))
    hdr = [str(h).split(" (")[0] if h else "" for h in rows[0]]
    return [dict(zip(hdr, r)) for r in rows[1:] if r[0] not in (None, "")]


def norm_name(s):
    s = s.replace("#U0130", "I")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.replace("ACCEESSORIES", "ACCESSORIES").lower()


def build_index(folder):
    index = {}
    for root, _, files in os.walk(folder):
        for f in files:
            index.setdefault(norm_name(f), os.path.join(root, f))
    if not index:
        raise FileNotFoundError(f"No files found under {folder}. Check the --photos path.")
    return index


def resolve(index, name):
    key = norm_name(name)
    if key not in index:
        raise FileNotFoundError(f"{name} not found. {len(index)} files were indexed, for example: {', '.join(list(index)[:3])}")
    return index[key]


def tr_lower(s):
    return s.replace("İ", "i").replace("I", "ı").lower()


def prep(text, keep_punct=False):
    t = tr_lower(str(text))
    if keep_punct:
        return t
    t = t.replace("â", "a").replace("î", "i").replace("û", "u")
    t = re.sub(r"[,.;:!?\"“”()]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_img(p):
    return (Image.open(p) if isinstance(p, str) else p).convert("RGB")


class SigLIP:
    def __init__(self, name, device=None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def _norm(self, x):
        if not self.torch.is_tensor(x):
            for attr in ("pooler_output", "image_embeds", "text_embeds"):
                v = getattr(x, attr, None)
                if v is not None:
                    x = v
                    break
            else:
                x = x[0]
        x = x / x.norm(dim=-1, keepdim=True)
        return x.cpu().numpy()

    def images(self, paths, item_ids=None, bs=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), bs):
                ims = [load_img(p) for p in paths[i:i + bs]]
                inp = self.proc(images=ims, return_tensors="pt").to(self.device)
                out.append(self._norm(self.model.get_image_features(**inp)))
        return np.concatenate(out)

    def texts(self, texts, item_ids=None, bs=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                inp = self.proc(text=texts[i:i + bs], padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
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


def nms_np(boxes, scores, iou_thr, k):
    order = np.argsort(-scores)
    keep = []
    while len(order) and len(keep) < k:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a_i + a_r - inter + 1e-9)
        order = rest[iou < iou_thr]
    return keep


class OwlDetector:
    """Class-agnostic object proposals from OWLv2's objectness head."""
    def __init__(self, name, device=None):
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(self.device).eval()

    def boxes(self, image, k=4, min_frac=0.02, iou_thr=0.4):
        W, H = image.size
        inp = self.proc(text=[["an object"]], images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        obj = out.objectness_logits[0].float().cpu().numpy()
        b = out.pred_boxes[0].float().cpu().numpy()          # cx, cy, w, h relative to the padded square
        S = max(W, H)
        xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * S, (b[:, 1] - b[:, 3] / 2) * S,
                         (b[:, 0] + b[:, 2] / 2) * S, (b[:, 1] + b[:, 3] / 2) * S], 1)
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H)
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        ok = np.where(area > min_frac * W * H)[0]
        if len(ok) == 0:
            return []
        keep = nms_np(xyxy[ok], obj[ok], iou_thr, k)
        return [tuple(xyxy[ok][i]) for i in keep]


class DummyDetector:
    def boxes(self, image, k=4, **kw):
        W, H = image.size
        return [(W * .1, H * .1, W * .6, H * .6), (W * .4, H * .4, W * .9, H * .9)][:k]


def crop_with_pad(img, box, pad=0.08):
    W, H = img.size
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def score_matrix(Q, G, q_items, g_items, hub=False, k=10):
    S = Q @ G.T
    if not hub:
        return S
    q_items = np.asarray(q_items); g_items = np.asarray(g_items)
    n = Q.shape[0]
    R = np.zeros_like(S)
    for j in range(G.shape[0]):
        col = S[:, j]
        idx = np.where(q_items != g_items[j])[0]
        order = idx[np.argsort(-col[idx])]
        top = order[:k + 1]
        for i in range(n):
            use = [t for t in top if t != i][:k]
            R[i, j] = col[use].mean() if use else 0.0
    return 2 * S - R


def rank_items(scores_row, g_items):
    best = {}
    for it, s in zip(g_items, scores_row):
        best[it] = max(best.get(it, -1e9), float(s))
    return sorted(best, key=lambda k: -best[k])


def summarize(rows, key, title):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r["rank"])
    print(f"\n{title}")
    print(f"{'':32}{'n':>5}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'MRR':>8}")
    for g in sorted(groups, key=str):
        rk = np.array(groups[g])
        print(f"{str(g):32}{len(rk):>5}" + "".join(f"{(rk <= k).mean():>8.2f}" for k in KS) + f"{(1 / rk).mean():>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--distractors", default=None)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--hub-norm", action="store_true")
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--save-crops", default=None)
    ap.add_argument("--keep-punct", action="store_true")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    wb = load_workbook(a.xlsx, data_only=True)
    items = {r["item_id"]: r for r in read_sheet(wb, "Items")}
    photos = read_sheet(wb, "Photos")
    descs = [d for d in read_sheet(wb, "Descriptions") if d.get("text_tr")]
    versions = [("raw", "text_tr")]
    if any(d.get("text_tr_clean") for d in descs):
        versions.append(("clean", "text_tr_clean"))
    norms = [("plain", False)] + ([("csls", True)] if a.hub_norm else [])
    modes = [("", False)] + ([("+crop", True)] if a.crop else [])
    print(f"{len(items)} items, {len(photos)} photos, {len(descs)} descriptions, text versions: {[v[0] for v in versions]}, "
          f"scoring: {[n[0] for n in norms]}, gallery: {['full'] + (['full+crops'] if a.crop else [])}")

    emb = Dummy() if a.dummy else SigLIP(a.model)
    index = build_index(a.photos)
    paths = {p["photo_file"]: resolve(index, p["photo_file"]) for p in photos}
    pf = [p["photo_file"] for p in photos]
    P = emb.images([paths[f] for f in pf], [p["item_id"] for p in photos])
    pvec = dict(zip(pf, P))

    dvec = {}
    for vname, col in versions:
        texts = [prep(d.get(col) or d["text_tr"], a.keep_punct) for d in descs]
        dvec[vname] = emb.texts(texts, [d["item_id"] for d in descs])
        if P.shape[1] != dvec[vname].shape[1]:
            raise SystemExit(f"Image vectors have {P.shape[1]} dimensions but text vectors have {dvec[vname].shape[1]}. "
                             "The model output was not read correctly; please send me this message.")
    print(f"Embeddings ready: {P.shape[1]} dimensions")

    if hasattr(emb, "proc"):
        try:
            ids = emb.proc.tokenizer([prep(d["text_tr"], a.keep_punct) for d in descs])["input_ids"]
            lens = [len(x) for x in ids]
            print(f"Text length check: {sum(1 for n in lens if n > 64)} of {len(lens)} descriptions are longer than 64 tokens (longest: {max(lens)}).")
            by_spec = collections.defaultdict(list)
            for d, n in zip(descs, lens):
                by_spec[d.get("specificity", "?")].append(n)
            print("  average tokens by specificity: " + ", ".join(f"{k} {np.mean(v):.0f}" for k, v in sorted(by_spec.items())))
        except Exception as e:
            print("Text length check skipped:", e)

    extra_vecs, extra_items = [], []
    if a.distractors:
        files = sorted(os.listdir(a.distractors))
        extra_vecs = list(emb.images([os.path.join(a.distractors, f) for f in files], [f"DIST_{i}" for i in range(len(files))]))
        extra_items = [f"DIST_{i}" for i in range(len(files))]
        print(f"{len(files)} distractor images added to every gallery")

    crop_vecs = {}
    if a.crop:
        det = DummyDetector() if a.dummy else OwlDetector(a.detector)
        if a.save_crops:
            os.makedirs(a.save_crops, exist_ok=True)
        n_crops = []
        for p in photos:
            if p["role"] not in ("F1", "F2"):
                continue
            img = load_img(paths[p["photo_file"]])
            crops = [c for c in (crop_with_pad(img, b) for b in det.boxes(img, k=a.crop)) if c is not None]
            n_crops.append(len(crops))
            if a.save_crops:
                stem = os.path.splitext(p["photo_file"])[0]
                for i, c in enumerate(crops):
                    c.save(os.path.join(a.save_crops, f"{stem}_crop{i}.jpg"))
            if crops:
                crop_vecs[p["photo_file"]] = emb.images(crops, [p["item_id"]] * len(crops))
        print(f"Crops: {np.mean(n_crops):.1f} per found photo on average (looked at {len(n_crops)} photos)")

    Lphotos = [p for p in photos if p["role"] == "L"]
    LV = np.stack([pvec[p["photo_file"]] for p in Lphotos])
    L_items = [p["item_id"] for p in Lphotos]
    d_items = [d["item_id"] for d in descs]

    rows = []
    for role in ("F1", "F2"):
        gal = [p for p in photos if p["role"] == role]
        sb = {p["item_id"]: p.get("same_background") for p in gal}
        for msuffix, use_crop in modes:
            vecs = [pvec[p["photo_file"]] for p in gal]
            g_items = [p["item_id"] for p in gal]
            if use_crop:
                for p in gal:
                    cv = crop_vecs.get(p["photo_file"])
                    if cv is not None:
                        vecs.extend(list(cv)); g_items.extend([p["item_id"]] * len(cv))
            G = np.stack(vecs + extra_vecs)
            g_items = g_items + extra_items

            def add(qtype, variant, norm, qid, item, srow, d=None):
                order = rank_items(srow, g_items)
                rows.append(dict(gallery=role, qtype=qtype, variant=variant, norm=norm, vn=f"{variant}/{norm}",
                                 desc_id=(d or {}).get("desc_id", ""), query=qid, item=item,
                                 category=items[item]["category"], same_background=sb.get(item),
                                 specificity=(d or {}).get("specificity", ""), dtype=(d or {}).get("type", ""),
                                 rank=order.index(item) + 1, top1=order[0], top3=" ".join(order[:3])))

            for nname, hub in norms:
                label = nname + msuffix
                for vname, col in versions:
                    S = score_matrix(dvec[vname], G, d_items, g_items, hub, a.hub_k)
                    for d, srow in zip(descs, S):
                        add("text", vname, label, d.get(col) or d["text_tr"], d["item_id"], srow, d)
                S = score_matrix(LV, G, L_items, g_items, hub, a.hub_k)
                for p, srow in zip(Lphotos, S):
                    add("photo", "photo", label, p["photo_file"], p["item_id"], srow)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n_gal = len(items) + len(extra_items)
    print(f"\nGallery size per run: {n_gal} items -> chance R@1 = {1 / n_gal:.2f}")
    for role in ("F1", "F2"):
        summarize([r for r in rows if r["qtype"] == "text" and r["gallery"] == role], "vn", f"text -> {role} gallery (text version / scoring)")
    for role in ("F1", "F2"):
        summarize([r for r in rows if r["qtype"] == "photo" and r["gallery"] == role], "vn", f"photo -> {role} gallery (scoring)")

    base = "raw/plain"
    txt = [r for r in rows if r["qtype"] == "text" and r["vn"] == base]
    summarize(txt, "category", f"text ({base}): by category (both galleries)")
    summarize(txt, "specificity", f"text ({base}): by specificity")
    summarize(txt, "dtype", f"text ({base}): by description type")
    if a.crop:
        best = "raw/plain+crop"
        summarize([r for r in rows if r["qtype"] == "text" and r["vn"] == best], "category", f"text ({best}): by category (both galleries)")

    print("\nHard Pass Rate (an item counts only if ALL its descriptions reach that rank)")
    for role in ("F1", "F2"):
        for vn in sorted({r["vn"] for r in rows if r["qtype"] == "text"}):
            per = collections.defaultdict(list)
            for r in rows:
                if r["qtype"] == "text" and r["gallery"] == role and r["vn"] == vn:
                    per[r["item"]].append(r["rank"])
            cells = "  ".join(f"@{k}: {np.mean([all(x <= k for x in v) for v in per.values()]):.2f}" for k in (1, 3, 5))
            print(f"  text -> {role}, {vn:18} {cells}")

    print("\nWrong top-1 winners (hub check): which gallery photo wins queries that belong to another item")
    for vn in sorted({r["vn"] for r in rows if r["qtype"] == "text"}):
        cnt = collections.Counter(r["top1"] for r in rows if r["qtype"] == "text" and r["vn"] == vn and r["rank"] > 1)
        shown = ", ".join(f"{k} x{v}" for k, v in cnt.most_common(4)) or "none"
        print(f"  {vn:18} {sum(cnt.values())} wrong top-1 in total; most frequent winners: {shown}")

    print("\nHardest target photos: ranks of the text queries (raw) and of the L-photo query. Look at these images.")
    for role in ("F1", "F2"):
        per = collections.defaultdict(lambda: {"text": [], "photo": None})
        for r in rows:
            if r["gallery"] != role:
                continue
            if r["qtype"] == "text" and r["vn"] == "raw/plain":
                per[r["item"]]["text"].append(r["rank"])
            if r["qtype"] == "photo" and r["vn"] == "photo/plain":
                per[r["item"]]["photo"] = r["rank"]
        worst = sorted(per.items(), key=lambda kv: -np.median(kv[1]["text"] + ([kv[1]["photo"]] if kv[1]["photo"] else [])))[:8]
        print(f"  {role} gallery:")
        for it, v in worst:
            print(f"    {it + '_' + role:22} text ranks {v['text']}  photo rank {v['photo']}")

    def compare(label, a_vn, b_vn, qtype, show=6):
        print(f"\n{label}")
        for role in ("F1", "F2"):
            A = {(r["desc_id"] or r["query"]): r for r in rows if r["gallery"] == role and r["vn"] == a_vn and r["qtype"] == qtype}
            B = {(r["desc_id"] or r["query"]): r for r in rows if r["gallery"] == role and r["vn"] == b_vn and r["qtype"] == qtype}
            diffs = [(A[k]["rank"], B[k]["rank"], k) for k in A if k in B and A[k]["rank"] != B[k]["rank"]]
            better = [d for d in diffs if d[1] < d[0]]; worse = [d for d in diffs if d[1] > d[0]]
            print(f"  {role}: {len(better)} better, {len(worse)} worse, {len(A) - len(diffs)} unchanged")
            for x, y, k in sorted(better, key=lambda t: t[1] - t[0])[:show]:
                print(f"    better  {k}: rank {x} -> {y}")
            for x, y, k in sorted(worse, key=lambda t: t[0] - t[1])[:show]:
                print(f"    worse   {k}: rank {x} -> {y}")

    if len(versions) > 1:
        compare("Corrected text vs raw text (plain scoring)", "raw/plain", "clean/plain", "text")
    if a.hub_norm:
        compare("Hub normalization vs plain (raw text)", "raw/plain", "raw/csls", "text")
        compare("Hub normalization vs plain (photo queries)", "photo/plain", "photo/csls", "photo")
    if a.crop:
        compare("Object crops vs full image only (raw text)", "raw/plain", "raw/plain+crop", "text")
        compare("Object crops vs full image only (photo queries)", "photo/plain", "photo/plain+crop", "photo")
    print(f"\nAll rows saved to {a.out}. Open it and sort by rank to see the mistakes.")


if __name__ == "__main__":
    main()
