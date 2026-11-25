"""Yeni (yapay zeka destekli) esyalar, eski esyalara gore daha mi kolay cikiyor?
Kullanim: python compare_new.py results.csv BAG_007 BAG_008 DOCS_007 ELEC_007 ELEC_008
results.csv: baseline_eval.py'nin urettigi dosya. Yeni esyalar ayri, digerleri ayri raporlanir."""
import csv, sys, collections
path, new = sys.argv[1], set(sys.argv[2:])
agg = collections.defaultdict(lambda: {"new": [], "old": []})
for r in csv.DictReader(open(path, encoding="utf-8")):
    k = (r["norm"], r["variant"], r["gallery"], r["qtype"])
    agg[k]["new" if r["item"] in new else "old"].append(int(float(r["rank"])))
print(f"{'norm':7}{'varyant':14}{'galeri':8}{'sorgu':8}{'n_yeni':>7}{'R@1 yeni':>10}{'n_eski':>7}{'R@1 eski':>10}")
for k in sorted(agg):
    a = agg[k]
    if not a["new"] or not a["old"]:
        continue
    f = lambda v: sum(x <= 1 for x in v) / len(v)
    print(f"{k[0]:7}{k[1]:14}{k[2]:8}{k[3]:8}{len(a['new']):>7}{f(a['new']):>10.2f}{len(a['old']):>7}{f(a['old']):>10.2f}")
print("\nYeni esyalar eskilerden BELIRGIN daha yuksekse (ornek: 1.00 vs 0.6) yapay gorseller fazla kolay demektir.")