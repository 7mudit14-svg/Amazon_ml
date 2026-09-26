"""Assemble the round-2 submission package in the official layout:
<team>_submission.zip
  output/matching_results.tsv, output/candidate_pairs.tsv
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md
src/ = the v10 pipeline copy used here (alt/v10p/src) + the alt add-ons (decision layer, tools).
"""
import argparse, os, shutil, zipfile
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="dir with matching_results.tsv and candidate_pairs.tsv")
ap.add_argument("--team", default="team")
ap.add_argument("--doc", required=True)
ap.add_argument("--out", default="/home/user/package")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(a.out, exist_ok=True)
z = os.path.join(a.out, f"{a.team}_submission.zip")
base = "code/business_entity_resolution"
with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as f:
    for n in ("matching_results.tsv", "candidate_pairs.tsv"):
        f.write(os.path.join(a.run, n), f"output/{n}")
    for d, arc in ((os.path.join(ROOT, "alt", "v10p", "src"), f"{base}/src"), (os.path.join(ROOT, "alt", "src"), f"{base}/src/alt")):
        for n in sorted(os.listdir(d)):
            if n.endswith(".py"):
                f.write(os.path.join(d, n), f"{arc}/{n}")
    f.write(os.path.join(ROOT, "alt", "PACKAGE_README.md"), f"{base}/README.md")
    f.write(os.path.join(ROOT, "v10", "code_v10", "code", "business_entity_resolution", "requirements.txt"), f"{base}/requirements.txt")
    f.write(a.doc, "Documentation_template.md")
print("wrote", z, os.path.getsize(z) // 2**20, "MB")
