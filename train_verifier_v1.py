"""
Ogrenilmis eslesme dogrulayici (verifier) v0: donmus SigLIP-2 (+ OWLv2 kirpmalari) ustune kucuk bir model.

Kurulum (bir kez):  pip install scikit-learn joblib      (baseline_eval.py'nin kurulumlari zaten yeterli)
Calistirma:         python train_verifier.py --xlsx DOSYA.xlsx --photos FOTOGRAF_KLASORU --cache ozellikler.joblib
                    (baseline_eval.py ayni klasorde olmali. Ilk calistirmada embedding'ler hesaplanir ve --cache dosyasina
                     kaydedilir, sonraki calistirmalar saniyeler surer.)
Deneme icin:        --dummy   (model indirmez; sadece kodun calistigini kontrol eder, sayilar anlamsizdir)

v1 farki: 'Eslesme yok' karari icin ayri, sorgu duzeyinde bir model eklendi (verifier_nomatch_*.joblib).
  Ozellikler: en yuksek skorlar, 1. ile 2./3. aday arasindaki fark, z-skoru, 1. adaya yakin aday sayisi.
  Etiket: sorgunun gercek esyasi galeride VAR (1) / galeriden cikarilmis YOK (0).

Ne yapar?
  Her (sorgu, aday esya) cifti icin birkac sayi cikarir (tam foto skoru, en iyi kirpma skoru, hub duzeltmeli skorlar,
  ilk aday ile fark, sira ...). Kucuk bir model (lojistik regresyon ve gradient boosting) bu sayilardan
  'bu cift gercekten esleserek mi' olasiligini ogrenir.
  Degerlendirme: esya bazli 5 parcali capraz dogrulama. Bir esyanin butun foto ve tarifleri (ve ikizi) ayni parcada olur,
  yani model test ettigi esyayi egitimde gormez.
  Kiyaslar: tam foto skoru, hub duzeltme+kirpma (elle ayarli en iyi yontem), LR, HGB.
  'Eslesme yok' testi: sorgunun gercek esyasi galeriden cikarilir; modelin en yuksek olasiligi ne kadar dusuk kalir?
  Sonunda butun veriyle bir son model kaydeder (verifier_*.joblib).
"""
import argparse, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_eval as be

FEATS = ["s_full", "s_crop", "s_best", "csls_full", "csls_crop", "csls_best", "z_best", "z_csls",
         "margin_best", "margin_csls", "inv_rank_best", "inv_rank_csls", "hub_full", "is_photo_query"]
IDX = {n: i for i, n in enumerate(FEATS)}


def hub_terms(bank, bank_items, G, g_items, k=10):
    """r_g = mean of the top-k similarities between gallery vector g and queries of OTHER items."""
    S = bank @ G.T
    bank_items = np.asarray(bank_items); g_items = np.asarray(g_items)
    r = np.zeros(G.shape[0])
    for j in range(G.shape[0]):
        col = S[bank_items != g_items[j], j]
        r[j] = np.sort(col)[-k:].mean() if len(col) else 0.0
    return r


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def _margin(x):
    order = np.sort(x)
    top1, top2 = order[-1], order[-2] if len(x) > 1 else order[-1]
    return np.where(x == top1, x - top2, x - top1)


def _inv_rank(x):
    return 1.0 / ((-x).argsort().argsort() + 1)


def make_feats(sf, sc, rf, rc, is_photo):
    best = np.maximum(sf, sc)
    csf, csc = 2 * sf - rf, 2 * sc - rc
    csb = np.maximum(csf, csc)
    return np.stack([sf, sc, best, csf, csc, csb, _z(best), _z(csb), _margin(best), _margin(csb),
                     _inv_rank(best), _inv_rank(csb), rf, np.full(len(sf), float(is_photo))], 1)


NMF = ["top1_raw", "top1_csls", "gap12_raw", "gap12_csls", "gap13_csls", "mean5_csls", "std_csls",
       "z1_csls", "n_close_csls", "n_close_raw", "is_photo"]
NIDX = {n: i for i, n in enumerate(NMF)}


def qfeat(sf, sc, rf, rc, is_photo):
    """Query-level features for the 'is the true item in the gallery at all?' decision."""
    raw = np.sort(np.maximum(sf, sc))[::-1]
    cs = np.sort(np.maximum(2 * sf - rf, 2 * sc - rc))[::-1]
    return np.array([raw[0], cs[0], raw[0] - raw[1], cs[0] - cs[1], cs[0] - cs[2], cs[:5].mean(), cs.std(),
                     (cs[0] - cs.mean()) / (cs.std() + 1e-9), float((cs > cs[0] - 0.05).sum()),
                     float((raw > raw[0] - 0.02).sum()), float(is_photo)])


def rank_of(scores, true_idx):
    return 1 + int(np.sum(scores > scores[true_idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--photos", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--detector", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--crop", type=int, default=4)
    ap.add_argument("--hub-k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--no-twin-heuristic", action="store_true")
    a = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import joblib

    from openpyxl import load_workbook
    wb = load_workbook(a.xlsx, data_only=True)
    items = {r["item_id"]: r for r in be.read_sheet(wb, "Items")}
    photos = be.read_sheet(wb, "Photos")
    descs = [d for d in be.read_sheet(wb, "Descriptions") if d.get("text_tr")]

    # ---- twin groups (cross-validation keeps twins together) ----
    group = {i: i for i in items}
    if any(r.get("twin") for r in items.values()):
        for i, r in items.items():
            t = r.get("twin")
            if t and t in items:
                group[i] = min(i, t); group[t] = min(i, t)
        print("Twin groups taken from the 'twin' column of the Items sheet.")
    elif not a.no_twin_heuristic:
        pairs = []
        for i in items:
            pre, num = i.rsplit("_", 1)
            if num == "001" and f"{pre}_006" in items:
                group[f"{pre}_006"] = i; pairs.append(f"{i}<->{pre}_006")
        print("No 'twin' column found. Heuristic used: NAME_001 and NAME_006 are twins ->", ", ".join(pairs) or "none")

    # ---- embeddings (cached) ----
    cache = None
    if a.cache and os.path.exists(a.cache):
        cache = joblib.load(a.cache)
        print(f"Loaded cached embeddings from {a.cache}")
    if cache is None:
        emb = be.Dummy() if a.dummy else be.SigLIP(a.model)
        index = be.build_index(a.photos)
        paths = {p["photo_file"]: be.resolve(index, p["photo_file"]) for p in photos}
        pf = [p["photo_file"] for p in photos]
        P = emb.images([paths[f] for f in pf], [p["item_id"] for p in photos])
        D = emb.texts([be.prep(d["text_tr"]) for d in descs], [d["item_id"] for d in descs])
        det = be.DummyDetector() if a.dummy else be.OwlDetector(a.detector)
        crops = {}
        for p in photos:
            if p["role"] not in ("F1", "F2"):
                continue
            img = be.load_img(paths[p["photo_file"]])
            cl = [c for c in (be.crop_with_pad(img, b) for b in det.boxes(img, k=a.crop)) if c is not None]
            if cl:
                crops[p["photo_file"]] = emb.images(cl, [p["item_id"]] * len(cl))
        cache = dict(pf=pf, P=P, D=D, crops=crops, desc_ids=[d["desc_id"] for d in descs])
        if a.cache:
            joblib.dump(cache, a.cache)
            print(f"Embeddings saved to {a.cache}")
    pvec = dict(zip(cache["pf"], cache["P"]))
    D = cache["D"]
    crops = cache["crops"]

    Lph = [p for p in photos if p["role"] == "L"]
    LV = np.stack([pvec[p["photo_file"]] for p in Lph]); L_items = [p["item_id"] for p in Lph]
    d_items = [d["item_id"] for d in descs]
    banks = {"text": (D, d_items), "photo": (LV, L_items)}

    # ---- build query records ----
    records = []
    for role in ("F1", "F2"):
        gal = [p for p in photos if p["role"] == role]
        cands = [p["item_id"] for p in gal]
        full = np.stack([pvec[p["photo_file"]] for p in gal])
        crop_list = [crops.get(p["photo_file"], np.zeros((0, full.shape[1]))) for p in gal]
        allc = np.concatenate(crop_list) if any(len(c) for c in crop_list) else np.zeros((0, full.shape[1]))
        owner = np.concatenate([[i] * len(c) for i, c in enumerate(crop_list)]).astype(int) if len(allc) else np.zeros(0, int)
        hub = {}
        for qt, (bank, bi) in banks.items():
            rf = hub_terms(bank, bi, full, cands, a.hub_k)
            rc = hub_terms(bank, bi, allc, [cands[o] for o in owner], a.hub_k) if len(allc) else np.zeros(0)
            hub[qt] = (rf, rc)
        queries = [("text", D[i], d_items[i]) for i in range(len(descs))] + \
                  [("photo", LV[i], L_items[i]) for i in range(len(Lph))]
        for qt, q, qitem in queries:
            sf = full @ q
            sc = sf.copy(); rcsel = hub[qt][0].copy()
            if len(allc):
                sall = allc @ q
                for ci in range(len(cands)):
                    m = owner == ci
                    if m.any():
                        j = np.argmax(sall[m]); sc[ci] = sall[m][j]; rcsel[ci] = hub[qt][1][m][j]
            records.append(dict(role=role, qtype=qt, qitem=qitem, cands=cands, sf=sf, sc=sc, rf=hub[qt][0], rc=rcsel,
                                X=make_feats(sf, sc, hub[qt][0], rcsel, qt == "photo")))
    print(f"{len(records)} queries built ({sum(len(r['cands']) for r in records)} query-candidate pairs)")

    # ---- cross-validation ----
    groups = sorted({group[r["qitem"]] for r in records})
    rng = np.random.default_rng(a.seed)
    folds = np.array_split(rng.permutation(groups), a.folds)
    grp_of_cand = lambda c: group[c]

    def new_models():
        return {"LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
                "HGB": HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                                      class_weight="balanced", random_state=a.seed)}

    res = collections.defaultdict(list)        # (qtype, role, method) -> ranks
    present, absent = collections.defaultdict(list), collections.defaultdict(list)
    nm_p, nm_a, nm_t = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for fi, test_groups in enumerate(folds):
        tg = set(test_groups.tolist())
        Xtr, ytr = [], []
        for r in records:
            if group[r["qitem"]] in tg:
                continue
            keep = np.array([group[c] not in tg for c in r["cands"]])
            Xtr.append(r["X"][keep]); ytr.append(np.array([c == r["qitem"] for c in r["cands"]])[keep].astype(int))
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        models = new_models()
        for m in models.values():
            m.fit(Xtr, ytr)
        Xq, yq = [], []
        for r in records:
            if group[r["qitem"]] in tg:
                continue
            ka = np.array([c != r["qitem"] for c in r["cands"]]); ip = r["qtype"] == "photo"
            Xq.append(qfeat(r["sf"], r["sc"], r["rf"], r["rc"], ip)); yq.append(1)
            Xq.append(qfeat(r["sf"][ka], r["sc"][ka], r["rf"][ka], r["rc"][ka], ip)); yq.append(0)
        Xq, yq = np.array(Xq), np.array(yq)
        nm_models = new_models()
        for m in nm_models.values():
            m.fit(Xq, yq)
        for r in records:
            if group[r["qitem"]] not in tg:
                continue
            ti = r["cands"].index(r["qitem"])
            tag = (r["qtype"], r["role"])
            scores = {"full cosine": r["X"][:, IDX["s_full"]], "hub+crop (elle ayarli)": r["X"][:, IDX["csls_best"]]}
            for n, m in models.items():
                scores[n] = m.predict_proba(r["X"])[:, 1]
            for n, s in scores.items():
                res[tag + (n,)].append(rank_of(s, ti))
            # simulated 'no match': remove the true item from the gallery and recompute the features
            keep = np.array([c != r["qitem"] for c in r["cands"]])
            Xa = make_feats(r["sf"][keep], r["sc"][keep], r["rf"][keep], r["rc"][keep], r["qtype"] == "photo")
            absent_scores = {"hub+crop (elle ayarli)": Xa[:, IDX["csls_best"]]}
            present_scores = {"hub+crop (elle ayarli)": r["X"][:, IDX["csls_best"]]}
            for n, m in models.items():
                absent_scores[n] = m.predict_proba(Xa)[:, 1]; present_scores[n] = scores[n]
            for n in absent_scores:
                present[n].append(present_scores[n].max()); absent[n].append(absent_scores[n].max())
            # v1: query-level no-match model
            ip = r["qtype"] == "photo"
            qp = qfeat(r["sf"], r["sc"], r["rf"], r["rc"], ip)
            qa = qfeat(r["sf"][keep], r["sc"][keep], r["rf"][keep], r["rc"][keep], ip)
            nm_scores = {"ham skor (top1 cosine)": (qp[NIDX["top1_raw"]], qa[NIDX["top1_raw"]]),
                         "hub+kirpma top1": (qp[NIDX["top1_csls"]], qa[NIDX["top1_csls"]]),
                         "1.-2. aday farki": (qp[NIDX["gap12_csls"]], qa[NIDX["gap12_csls"]]),
                         "z-skoru (1. aday)": (qp[NIDX["z1_csls"]], qa[NIDX["z1_csls"]])}
            for n, m in nm_models.items():
                nm_scores["v1 " + n] = (m.predict_proba(qp[None])[0, 1], m.predict_proba(qa[None])[0, 1])
            for n, (vp, va) in nm_scores.items():
                nm_p[n].append(vp); nm_a[n].append(va); nm_t[n].append(ip)

    methods = ["full cosine", "hub+crop (elle ayarli)", "LR", "HGB"]
    print("\n=== Esya bazli capraz dogrulama (her sorgu bir kez, hic gormedigi esyayla test edilir) ===")
    for qt in ("text", "photo"):
        for role in ("F1", "F2"):
            print(f"\n{qt} -> {role} galerisi")
            print(f"{'yontem':28}{'n':>5}{'R@1':>8}{'R@3':>8}{'MRR':>8}")
            for m in methods:
                rk = np.array(res[(qt, role, m)])
                print(f"{m:28}{len(rk):>5}{(rk <= 1).mean():>8.2f}{(rk <= 3).mean():>8.2f}{(1 / rk).mean():>8.2f}")
    print("\nHepsi birlikte (text + photo, F1 + F2)")
    print(f"{'yontem':28}{'n':>5}{'R@1':>8}{'R@3':>8}{'MRR':>8}")
    for m in methods:
        rk = np.concatenate([np.array(v) for k, v in res.items() if k[2] == m])
        print(f"{m:28}{len(rk):>5}{(rk <= 1).mean():>8.2f}{(rk <= 3).mean():>8.2f}{(1 / rk).mean():>8.2f}")

    print("\n=== 'Eslesme yok' testi ===")
    print("Her sorgu iki kez sorulur: esyasi galeride VAR ve YOK. Model en yuksek skorla ayirt edebiliyor mu?")
    print("AUC 1.0 mukemmel, 0.5 rastgele. 'Yanlis eslesme' = esyasi galeride olmayan sorguda esik ustu skor cikmasi.")
    print(f"{'yontem':28}{'AUC':>8}{'yanlis eslesme @%90 dogru kabul':>36}")
    for n in ("hub+crop (elle ayarli)", "LR", "HGB"):
        p, q = np.array(present[n]), np.array(absent[n])
        y = np.r_[np.ones(len(p)), np.zeros(len(q))]
        auc = roc_auc_score(y, np.r_[p, q])
        thr = np.quantile(p, 0.10)
        print(f"{n:28}{auc:>8.2f}{(q >= thr).mean():>36.2f}")
    print("(Hub+crop skoru olasilik degil, birimsiz skor; AUC yine de karsilastirilabilir.)")

    print("\n=== v1: 'Eslesme yok' karari (sorgu duzeyinde model) ===")
    print("AUC 1.0 mukemmel, 0.5 rastgele. 'Yanlis eslesme' = esyasi galeride OLMAYAN sorguda sistemin 'buldum' demesi,")
    print("esik, esyasi galeride OLAN sorgularin %90 (veya %80) inin kabul edilecegi sekilde secilir.")
    print(f"{'yontem':28}{'AUC':>7}{'AUC metin':>11}{'AUC foto':>10}{'YE@%90':>9}{'YE@%80':>9}")
    for n in nm_p:
        p, q, t = np.array(nm_p[n]), np.array(nm_a[n]), np.array(nm_t[n])
        y = np.r_[np.ones(len(p)), np.zeros(len(q))]
        auc = roc_auc_score(y, np.r_[p, q])
        aucs = []
        for ph in (False, True):
            mk = t == ph
            aucs.append(roc_auc_score(np.r_[np.ones(mk.sum()), np.zeros(mk.sum())], np.r_[p[mk], q[mk]]))
        fm = lambda keep: (q >= np.quantile(p, 1 - keep)).mean()
        print(f"{n:28}{auc:>7.2f}{aucs[0]:>11.2f}{aucs[1]:>10.2f}{fm(0.9):>9.2f}{fm(0.8):>9.2f}")
    print("YE = yanlis eslesme orani (dusuk iyi). Uygulamada esigi, kullaniciya yanlis aday gostermemek icin siki sececegiz.")

    # ---- final models on all data ----
    Xall = np.concatenate([r["X"] for r in records])
    yall = np.concatenate([(np.array(r["cands"]) == r["qitem"]).astype(int) for r in records])
    out_dir = os.path.dirname(os.path.abspath(a.cache)) if a.cache else "."
    for n, m in new_models().items():
        m.fit(Xall, yall)
        path = os.path.join(out_dir, f"verifier_{n.lower()}.joblib")
        joblib.dump(dict(model=m, features=FEATS), path)
        print(f"Son model kaydedildi: {path}")
    Xq, yq = [], []
    for r in records:
        ka = np.array([c != r["qitem"] for c in r["cands"]]); ip = r["qtype"] == "photo"
        Xq.append(qfeat(r["sf"], r["sc"], r["rf"], r["rc"], ip)); yq.append(1)
        Xq.append(qfeat(r["sf"][ka], r["sc"][ka], r["rf"][ka], r["rc"][ka], ip)); yq.append(0)
    for n, m in new_models().items():
        m.fit(np.array(Xq), np.array(yq))
        path = os.path.join(out_dir, f"verifier_nomatch_{n.lower()}.joblib")
        joblib.dump(dict(model=m, features=NMF), path)
        print(f"Eslesme-yok modeli kaydedildi: {path}")
    print("\nNOT: Bu v1. Hub terimi sorgu bankasindan hesaplaniyor (etiket kullanilmiyor ama sorgular ortak). "
          "Nihai test setinde bu sizinti olmayacak sekilde yeniden hesaplanacak.")


if __name__ == "__main__":
    main()