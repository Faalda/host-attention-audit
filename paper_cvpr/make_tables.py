"""Generate LaTeX tables for the CVPR revision directly from result JSONs (no hand transcription).
Usage: ../.venv/bin/python3 paper_cvpr/make_tables.py  -> writes paper_cvpr/gen_tables.tex"""
import json, os

R = "/home/faldabbas/myproject/attn_diffusion_gen/results/revision"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_tables.tex")
NAMES = {"free": "Free plan", "onehot": "On-manifold plan", "hidden": "Hidden-state bias",
         "full": "Full-rank bias", "rank2": "Rank-2 bias", "query": "Query-embedding offset",
         "const": "Learned constant bias", "img": "Learned image-dependent bias"}
PROTO = [("single_ref", "Single-ref"), ("heldout", "Held-out refs"), ("mismatch_target", "Shuffled: fit"),
         ("mismatch_grounding", "Shuffled: grounding")]


def fmt(m, k, scale=1.0, nd=3):
    x = m[k]
    return (f"{x['inv'] * scale:.{nd}f} & ${x['delta'] * scale:+.{nd}f}$ "
            f"{{\\scriptsize[{x['ci95'][0] * scale:+.{nd}f}, {x['ci95'][1] * scale:+.{nd}f}]}}")


def full_metric_table(path, label, caption, protocols=PROTO):
    if not os.path.exists(path):
        return f"% missing {path}\n"
    d = json.load(open(path))
    vs = [k for k in d if k not in ("file", "n_images")]
    lines = [r"\begin{table*}[t]\centering", f"\\caption{{{caption}}}\\label{{{label}}}", r"\small",
             r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{ll" + "cc" * 4 + "}", r"\toprule",
             r"& & \multicolumn{2}{c}{Standard ROUGE-1} & \multicolumn{2}{c}{ROUGE-L} & \multicolumn{2}{c}{BLEU-4} & \multicolumn{2}{c}{CIDEr-D} \\",
             r"Variant & Protocol & oracle & $\Delta$ & oracle & $\Delta$ & oracle & $\Delta$ & oracle & $\Delta$ \\", r"\midrule"]
    for v in vs:
        for i, (pk, pn) in enumerate(protocols):
            m = d[v][pk]
            row = (NAMES.get(v, v) if i == 0 else "") + f" & {pn} & " + " & ".join(
                fmt(m, k, nd=4 if k == "rouge1" else 3)
                for k in ["rouge1", "rougeL", "bleu4", "cider"]) + r" \\"
            lines.append(row)
        h = d[v]
        shuffled = (f"shuffled {h['halluc_mismatch_inv']['chair_s_proxy']:.3f}, "
                    if any(pk == "mismatch_target" for pk, _ in protocols) else "")
        lines.append(f" & \\multicolumn{{9}}{{l}}{{\\scriptsize Hallucination proxy: true target {h['halluc_true_inv']['chair_s_proxy']:.3f}, "
                     f"{shuffled}baseline {h['halluc_baseline']['chair_s_proxy']:.3f}; "
                     f"exact match with the target {h['single_ref']['exact']['inv']:.3f} vs {h['single_ref']['exact']['base']:.3f}}} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}", ""]
    return "\n".join(lines)


SNAME = {"top": "Top-$K$ mass", "token": "Token-conditioned", "rows": "Matched rows", "random": "Random patches", "bottom": "Bottom-$K$"}
TNAME = {"recipient baseline": "Recipient, own plan", "donor own caption (X_j)": "Donor image, own plan",
         "transplant:donor": "Donor plan", "transplant:permuted": "Donor plan, permuted patches",
         "transplant:rowshuf": "Donor plan, shuffled rows", "transplant:uniform": "Uniform plan"}


def interventions_tables(path):
    if not os.path.exists(path):
        return f"% missing {path}\n"
    d = json.load(open(path))
    s = [r"\begin{table}[t]\centering", r"\caption{Ancillary suppression with $K\in\{3,10\}$ on 500 test images, measured against the unedited guided caption. Similarity is legacy set-unigram F1; rows count steps losing the external plan bias, not native query--key attention.}",
         r"\label{tab:supp_full}\small", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Strategy & Rows & Changed & Primary removed & Any object removed & Sim. \\", r"\midrule"]
    order = ["top", "token", "rows", "random", "bottom"]
    for K in (3, 10):
        for o in order:
            k = f"{o}_K{K}"
            if k not in d["suppression"]:
                continue
            m = d["suppression"][k]
            s.append(f"{SNAME[o]} ($K{{=}}{K}$) & {m['rows_unrouted_mean']:.1f} & {m['changed_pct']:.1f}\\% & {m['primary_removed_pct']:.1f}\\% & "
                     f"{m['any_obj_removed_pct']:.1f}\\% & {m['sim_to_unedited']:.3f} \\\\")
        s.append(r"\midrule")
    s[-1] = r"\bottomrule"
    s += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    t = [r"\begin{table}[t]\centering", r"\caption{Legacy ancillary plan transplant (500 recipient--donor pairs). Best-of-five legacy set-unigram ROUGE-1 and CIDEr-D against recipient and donor references; \emph{donor-only obj.}: fraction of mentioned COCO objects that occur only in donor references. Uniform plan removes only the external prior, retaining native learned query--key attention.}",
         r"\label{tab:transplant_full}\small", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lcccccc}", r"\toprule",
          r"Condition & Legacy R1 rec. & Legacy R1 don. & CIDEr rec. & CIDEr don. & donor-only obj. & closer to donor \\", r"\midrule"]
    for name, m in d["transplant"].items():
        t.append(f"{TNAME.get(name, name)} & {m['r1_recipient']:.3f} & {m['r1_donor']:.3f} & {m['cider_recipient']:.3f} & "
                 f"{m['cider_donor']:.3f} & {m['donor_object_rate']:.3f} & {m['frac_closer_to_donor']:.3f} \\\\")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(s) + "\n" + "\n".join(t)


def main():
    parts = [
        full_metric_table(f"{R}/inversion_protocol_n1000_s0_metrics.json", "tab:e1_full",
                          "All metrics for the small captioner ($n{=}1000$). Paired differences from the denoised-plan baseline with 95\\% bootstrap CIs over images."),
        full_metric_table(f"{R}/blip2_protocol_n500_s0_metrics.json", "tab:e2_full",
                          "All metrics for BLIP-2 ($n{=}500$); the baseline is BLIP-2 without bias."),
        full_metric_table(f"{R}/blip2_forward_n500_r8_s0_metrics.json", "tab:e3_full",
                           "Learned forward predictors for BLIP-2, which see no test references, scored on the same reference slots as the inversion results. Dummy shuffled-target fields are not evaluated.",
                           protocols=PROTO[:2]),
        interventions_tables(f"{R}/interventions_n500_s0_summary.json"),
    ]
    open(OUT, "w").write("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
