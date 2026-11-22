"""
Baseline for the lost-item pilot: SigLIP-2 matching, no training.

Setup (once):   pip install torch transformers pillow numpy openpyxl
Run:            python baseline_eval.py --xlsx lost_item_pilot_dataset_duzeltilmis.xlsx --photos PATH\\TO\\photos
Optional:       --hub-norm     also run with hub normalization (CSLS) and compare it with the plain scores
                --hub-k 10     how many neighbours CSLS averages (default 10)
                --distractors PATH\\TO\\extra_images   (extra gallery images; added to every gallery)
                --model google/siglip2-base-patch16-224   (default)
                --keep-punct   (do not strip commas and capital letters before embedding)
                --dummy        (no model; only checks that the data and metrics code work)

What it measures (item level; chance = 1/N for N gallery items):
  text -> photo : each Turkish description searches the F1 gallery and the F2 gallery.
                  If the sheet has a text_tr_clean column, BOTH versions are run (spelling cost).
  photo -> photo: each L photo searches the F1 gallery and the F2 gallery.
Metrics: Recall@1/3/5/10, MRR, Hard Pass Rate@1/3/5, and a 'wrong top-1 winners' table that shows hub items.

Hub normalization (CSLS): some gallery photos look 'a bit similar to everything' and win vague queries.
CSLS lowers such a photo's score by how strongly it matches typical queries:
    score(q, g) = 2 * cos(q, g) - r_g,   r_g = mean of the top-k cosines between g and a bank of other queries.
The bank is the other descriptions in the sheet (for photo queries: the other L photos). To stay fair, the bank
never contains the query itself or any description of the gallery photo's own item.
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

    def token_lengths(self, texts):
        tok = getattr(self.proc, "tokenizer", None)
        if tok is None:
            return None
        return [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]

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


def score_matrix(Q, G, q_items, g_items, hub=False, k=10):
    """Cosine scores (queries x gallery photos). With hub=True, apply CSLS using the queries themselves as the bank."""
    S = Q @ G.T
    if not hub:
        return S
    q_items = np.asarray(q_items)
    g_items = np.asarray(g_items)
    n = Q.shape[0]
    R = np.zeros_like(S)
    for j in range(G.shape[0]):
        col = S[:, j]
        idx = np.where(q_items != g_items[j])[0]          # bank: never the gallery photo's own item
        order = idx[np.argsort(-col[idx])]
        top = order[:k + 1]
        for i in range(n):
            use = [t for t in top if t != i][:k]           # never the query itself
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
        line = f"{str(g):32}{len(rk):>5}" + "".join(f"{(rk <= k).mean():>8.2f}" for k in KS)
        print(line + f"{(1 / rk).mean():>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--distractors", default=None)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--hub-norm", action="store_true")
    ap.add_argument("--hub-k", type=int, default=10)
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
    print(f"{len(items)} items, {len(photos)} photos, {len(descs)} descriptions, "
          f"text versions: {[v[0] for v in versions]}, scoring: {[n[0] for n in norms]}")

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
        if hasattr(emb, "token_lengths"):
            try:
                lens = emb.token_lengths(texts)
                if lens:
                    over = [i for i, l in enumerate(lens) if l > 64]
                    print(f"Text length ({vname}): longest {max(lens)} tokens; {len(over)} of {len(lens)} descriptions "
                          f"exceed the model's 64-token limit and are cut off")
                    if over:
                        by_spec = collections.Counter(descs[i].get("specificity") for i in over)
                        print(f"  cut-off descriptions by specificity: {dict(by_spec)}")
            except Exception as e:  # never stop the run because of this check
                print(f"(token length check skipped: {e})")
        if P.shape[1] != dvec[vname].shape[1]:
            raise SystemExit(f"Image vectors have {P.shape[1]} dimensions but text vectors have {dvec[vname].shape[1]}. "
                             "The model output was not read correctly; please send me this message.")
    print(f"Embeddings ready: {P.shape[1]} dimensions")

    if hasattr(emb, "proc"):
        try:
            ids = emb.proc.tokenizer([prep(d["text_tr"], a.keep_punct) for d in descs])["input_ids"]
            lens = [len(x) for x in ids]
            cut = sum(1 for n in lens if n > 64)
            print(f"Text length check: {cut} of {len(lens)} descriptions are longer than 64 tokens and get cut off "
                  f"(longest: {max(lens)}). The text encoder only reads the first 64 tokens.")
            by_spec = collections.defaultdict(list)
            for d, n in zip(descs, lens):
                by_spec[d.get("specificity", "?")].append(n)
            print("  average tokens by specificity: " + ", ".join(f"{k} {np.mean(v):.0f}" for k, v in sorted(by_spec.items())))
        except Exception as e:  # never stop the run because of this check
            print("Text length check skipped:", e)

    extra_vecs, extra_items = [], []
    if a.distractors:
        files = sorted(os.listdir(a.distractors))
        extra_vecs = list(emb.images([os.path.join(a.distractors, f) for f in files], [f"DIST_{i}" for i in range(len(files))]))
        extra_items = [f"DIST_{i}" for i in range(len(files))]
        print(f"{len(files)} distractor images added to every gallery")

    Lphotos = [p for p in photos if p["role"] == "L"]
    LV = np.stack([pvec[p["photo_file"]] for p in Lphotos])
    L_items = [p["item_id"] for p in Lphotos]
    d_items = [d["item_id"] for d in descs]

    rows = []
    for role in ("F1", "F2"):
        gal = [p for p in photos if p["role"] == role]
        G = np.stack([pvec[p["photo_file"]] for p in gal] + extra_vecs)
        g_items = [p["item_id"] for p in gal] + extra_items
        sb = {p["item_id"]: p.get("same_background") for p in gal}

        def add(qtype, variant, norm, qid, item, srow, d=None):
            order = rank_items(srow, g_items)
            rows.append(dict(gallery=role, qtype=qtype, variant=variant, norm=norm, vn=f"{variant}/{norm}",
                             desc_id=(d or {}).get("desc_id", ""), query=qid, item=item,
                             category=items[item]["category"], same_background=sb.get(item),
                             specificity=(d or {}).get("specificity", ""), dtype=(d or {}).get("type", ""),
                             rank=order.index(item) + 1, top1=order[0], top3=" ".join(order[:3])))

        for nname, hub in norms:
            for vname, col in versions:
                S = score_matrix(dvec[vname], G, d_items, g_items, hub, a.hub_k)
                for d, srow in zip(descs, S):
                    add("text", vname, nname, d.get(col) or d["text_tr"], d["item_id"], srow, d)
            S = score_matrix(LV, G, L_items, g_items, hub, a.hub_k)
            for p, srow in zip(Lphotos, S):
                add("photo", "photo", nname, p["photo_file"], p["item_id"], srow)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    n_gal = len(items) + len(extra_items)
    print(f"\nGallery size per run: {n_gal} items -> chance R@1 = {1 / n_gal:.2f}")
    for role in ("F1", "F2"):
        sub = [r for r in rows if r["qtype"] == "text" and r["gallery"] == role]
        summarize(sub, "vn", f"text -> {role} gallery (text version / scoring)")
    for role in ("F1", "F2"):
        sub = [r for r in rows if r["qtype"] == "photo" and r["gallery"] == role]
        summarize(sub, "vn", f"photo -> {role} gallery (scoring)")

    base = "raw/plain"
    txt = [r for r in rows if r["qtype"] == "text" and r["vn"] == base]
    summarize(txt, "category", f"text ({base}): by category (both galleries)")
    summarize(txt, "specificity", f"text ({base}): by specificity")
    summarize(txt, "dtype", f"text ({base}): by description type")

    print("\nHard Pass Rate (an item counts only if ALL its descriptions reach that rank)")
    for role in ("F1", "F2"):
        for vn in sorted({r["vn"] for r in rows if r["qtype"] == "text"}):
            per = collections.defaultdict(list)
            for r in rows:
                if r["qtype"] == "text" and r["gallery"] == role and r["vn"] == vn:
                    per[r["item"]].append(r["rank"])
            cells = "  ".join(f"@{k}: {np.mean([all(x <= k for x in v) for v in per.values()]):.2f}" for k in (1, 3, 5))
            print(f"  text -> {role}, {vn:12} {cells}")

    print("\nWrong top-1 winners (hub check): which gallery photo wins queries that belong to another item")
    for vn in sorted({r["vn"] for r in rows if r["qtype"] == "text"}):
        cnt = collections.Counter(r["top1"] for r in rows if r["qtype"] == "text" and r["vn"] == vn and r["rank"] > 1)
        tot = sum(cnt.values())
        shown = ", ".join(f"{k} x{v}" for k, v in cnt.most_common(4)) or "none"
        print(f"  {vn:12} {tot} wrong top-1 in total; most frequent winners: {shown}")

    def compare(label, a_vn, b_vn, qtype):
        print(f"\n{label}")
        for role in ("F1", "F2"):
            A = {(r["desc_id"] or r["query"]): r for r in rows if r["gallery"] == role and r["vn"] == a_vn and r["qtype"] == qtype}
            B = {(r["desc_id"] or r["query"]): r for r in rows if r["gallery"] == role and r["vn"] == b_vn and r["qtype"] == qtype}
            diffs = [(A[k]["rank"], B[k]["rank"], k) for k in A if k in B and A[k]["rank"] != B[k]["rank"]]
            better = sum(1 for x, y, _ in diffs if y < x); worse = sum(1 for x, y, _ in diffs if y > x)
            print(f"  {role}: {better} better, {worse} worse, {len(A) - len(diffs)} unchanged")
            for x, y, k in sorted(diffs, key=lambda t: t[1] - t[0]):
                print(f"    {k}: rank {x} -> {y}")

    print("\nHardest target photos: median rank over the text queries (raw) and the L-photo query. Look at these images.")
    for role in ("F1", "F2"):
        per = collections.defaultdict(lambda: {"text": [], "photo": None})
        for r in rows:
            if r["gallery"] != role:
                continue
            if r["qtype"] == "text" and r["vn"] == "raw/plain":
                per[r["item"]]["text"].append(r["rank"])
            if r["qtype"] == "photo" and r["norm"] == "plain":
                per[r["item"]]["photo"] = r["rank"]
        worst = sorted(per.items(), key=lambda kv: -np.median(kv[1]["text"] + ([kv[1]["photo"]] if kv[1]["photo"] else [])))[:8]
        print(f"  {role} gallery:")
        for it, v in worst:
            print(f"    {it + '_' + role:22} text ranks {v['text']}  photo rank {v['photo']}")

    if len(versions) > 1:
        compare("Queries where the corrected text changed the rank (plain scoring)", "raw/plain", "clean/plain", "text")
    if a.hub_norm:
        compare("Queries where hub normalization changed the rank (raw text)", "raw/plain", "raw/csls", "text")
        compare("Photo queries where hub normalization changed the rank", "photo/plain", "photo/csls", "photo")
    print(f"\nAll rows saved to {a.out}. Open it and sort by rank to see the mistakes.")


if __name__ == "__main__":
    main()