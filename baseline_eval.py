"""
Baseline for the lost-item pilot: SigLIP-2 matching, no training.

Setup (once):   pip install torch transformers pillow numpy openpyxl
Run:            python baseline_eval.py --xlsx lost_item_pilot_dataset_duzeltilmis.xlsx --photos PATH\\TO\\photos
Optional:       --distractors PATH\\TO\\extra_images   (extra gallery images; added to every gallery)
                --model google/siglip2-base-patch16-224   (default)
                --keep-punct   (do not strip commas and capital letters before embedding)
                --dummy        (no model; only checks that the data and metrics code work)

What it measures (item level; chance = 1/N for N gallery items):
  text -> photo : each Turkish description searches the F1 gallery and the F2 gallery.
                  If the sheet has a text_tr_clean column, BOTH versions are run, so you see what spelling mistakes cost.
  photo -> photo: each L photo searches the F1 gallery and the F2 gallery.
Texts are lower-cased and stripped of commas/circumflexes before embedding (for both versions), so the raw-vs-clean
difference reflects spelling, not punctuation. Use --keep-punct to switch this off.
Metrics: Recall@1/3/5/10, MRR, Hard Pass Rate@1/3/5 (an item counts only if all its descriptions reach that rank).
Everything is saved to results.csv.
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
    s = s.replace("#U0130", "I")  # some unzip tools write the dotted capital I like this
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.replace("ACCEESSORIES", "ACCESSORIES").lower()


def build_index(folder):
    """Index every file under the folder (sub-folders included) by a normalized name."""
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
        sample = ", ".join(list(index)[:3])
        raise FileNotFoundError(f"{name} not found. {len(index)} files were indexed, for example: {sample}")
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


class SigLIP:
    def __init__(self, name, device=None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def _norm(self, x):
        # Newer transformers versions return an output object instead of a tensor.
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
                ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
                inp = self.proc(images=ims, return_tensors="pt").to(self.device)
                out.append(self._norm(self.model.get_image_features(**inp)))
        return np.concatenate(out)

    def texts(self, texts, item_ids=None, bs=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                inp = self.proc(text=texts[i:i + bs], padding="max_length",
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
    print(f"{'':32}{'n':>5}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'MRR':>8}")
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
    print(f"{len(items)} items, {len(photos)} photos, {len(descs)} descriptions, text versions: {[v[0] for v in versions]}")

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

    extra_vecs, extra_items = [], []
    if a.distractors:
        files = sorted(os.listdir(a.distractors))
        extra_vecs = list(emb.images([os.path.join(a.distractors, f) for f in files], [f"DIST_{i}" for i in range(len(files))]))
        extra_items = [f"DIST_{i}" for i in range(len(files))]
        print(f"{len(files)} distractor images added to every gallery")

    rows = []
    for role in ("F1", "F2"):
        gal = [p for p in photos if p["role"] == role]
        G = np.stack([pvec[p["photo_file"]] for p in gal] + extra_vecs)
        g_items = [p["item_id"] for p in gal] + extra_items
        sb = {p["item_id"]: p.get("same_background") for p in gal}

        def add(qtype, variant, qid, item, vec, d=None):
            order = item_ranking(vec, G, g_items)
            rows.append(dict(gallery=role, qtype=qtype, variant=variant, desc_id=(d or {}).get("desc_id", ""),
                             query=qid, item=item, category=items[item]["category"], same_background=sb.get(item),
                             specificity=(d or {}).get("specificity", ""), dtype=(d or {}).get("type", ""),
                             rank=order.index(item) + 1, top1=order[0], top3=" ".join(order[:3])))

        for vname, col in versions:
            for d, v in zip(descs, dvec[vname]):
                add("text", vname, d.get(col) or d["text_tr"], d["item_id"], v, d)
        for p in photos:
            if p["role"] == "L":
                add("photo", "photo", p["photo_file"], p["item_id"], pvec[p["photo_file"]])

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n_gal = len(items) + len(extra_items)
    print(f"\nGallery size per run: {n_gal} items -> chance R@1 = {1 / n_gal:.2f}")
    for role in ("F1", "F2"):
        sub = [r for r in rows if r["qtype"] == "text" and r["gallery"] == role]
        summarize(sub, "variant", f"text -> {role} gallery, by text version")
    for role in ("F1", "F2"):
        sub = [dict(r, all="all") for r in rows if r["qtype"] == "photo" and r["gallery"] == role]
        summarize(sub, "all", f"photo -> {role} gallery")

    base = "raw"
    txt = [r for r in rows if r["qtype"] == "text" and r["variant"] == base]
    summarize(txt, "category", f"text ({base}): by category (both galleries)")
    summarize(txt, "specificity", f"text ({base}): by specificity")
    summarize(txt, "dtype", f"text ({base}): by description type")
    ph = [r for r in rows if r["qtype"] == "photo"]
    summarize(ph, "same_background", "photo -> photo: by same_background of the target photo (text queries are not affected by background)")

    print("\nHard Pass Rate (an item counts only if ALL its descriptions reach that rank)")
    for role in ("F1", "F2"):
        for vname, _ in versions:
            per = collections.defaultdict(list)
            for r in rows:
                if r["qtype"] == "text" and r["gallery"] == role and r["variant"] == vname:
                    per[r["item"]].append(r["rank"])
            cells = "  ".join(f"@{k}: {np.mean([all(x <= k for x in v) for v in per.values()]):.2f}" for k in (1, 3, 5))
            print(f"  text -> {role}, {vname:5}  {cells}")

    if len(versions) > 1:
        print("\nQueries where the corrected text changed the rank")
        for role in ("F1", "F2"):
            raw = {r["desc_id"]: r for r in rows if r["gallery"] == role and r["variant"] == "raw"}
            cln = {r["desc_id"]: r for r in rows if r["gallery"] == role and r["variant"] == "clean"}
            diffs = [(raw[k]["rank"], cln[k]["rank"], k) for k in raw if raw[k]["rank"] != cln[k]["rank"]]
            better = sum(1 for x, y, _ in diffs if y < x); worse = sum(1 for x, y, _ in diffs if y > x)
            print(f"  {role}: {better} better, {worse} worse, {len(raw) - len(diffs)} unchanged")
            for x, y, k in sorted(diffs, key=lambda t: t[1] - t[0]):
                print(f"    {k}: rank {x} -> {y}")
    print(f"\nAll rows saved to {a.out}. Open it and sort by rank to see the mistakes.")


if __name__ == "__main__":
    main()
