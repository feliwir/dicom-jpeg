#!/usr/bin/env python3
"""Turn the Criterion results of the benchmarks into SVG charts and tables.

    cargo bench -p dicom-jpeg-benchmarks
    cargo bench -p dicom-jpeg-benchmarks 2> bench.log
    python3 benchmarks/plot.py [--log bench.log] [--criterion target/criterion] [--out benchmarks/results]

Writes one bar chart per codec (dicom-jpeg against the references), one for
the SIMD kernels (each level against scalar), and results/README.md with the
charts and the same numbers as tables. With `--log` (the benches' stderr),
encoders are labelled with their compression ratio. Only the standard
library is used.
"""

import argparse
import datetime
import html
import json
import os
import platform
import re
from pathlib import Path

LEVELS = ["scalar", "sse41", "avx2", "avx512", "neon"]
LEVEL_SUFFIX = re.compile(r" \((%s)\)$" % "|".join(LEVELS))

CODECS = [
    ("jpeg", "JPEG (T.81)"),
    ("jpegls", "JPEG-LS (T.87)"),
    ("j2k", "JPEG 2000"),
    ("htj2k", "HTJ2K"),
    ("jxl", "JPEG XL"),
]

# Colours: the reference data-viz palette. dicom-jpeg is the accent (blue),
# references the de-emphasis grey; kernel levels an ordinal blue ramp. Each
# role has a light and a dark value (prefers-color-scheme).
STYLE = """
.bg { fill: #fcfcfb; }
.title { fill: #0b0b0b; font: 600 17px system-ui, -apple-system, 'Segoe UI', sans-serif; }
.sub { fill: #52514e; font: 12.5px system-ui, -apple-system, 'Segoe UI', sans-serif; }
.group { fill: #0b0b0b; font: 600 13px system-ui, -apple-system, 'Segoe UI', sans-serif; }
.name { fill: #52514e; font: 12.5px system-ui, -apple-system, 'Segoe UI', sans-serif; }
.name.ours { fill: #0b0b0b; font-weight: 600; }
.value { fill: #52514e; font: 12px ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
.value.ours { fill: #0b0b0b; font-weight: 600; }
.axis { stroke: #c3c2b7; stroke-width: 1; }
.ours-fill { fill: #2a78d6; }
.ref-fill { fill: #9b9a94; }
.l0 { fill: #86b6ef; } .l1 { fill: #5598e7; } .l2 { fill: #256abf; } .l3 { fill: #184f95; } .l4 { fill: #104281; }
@media (prefers-color-scheme: dark) {
  .bg { fill: #1a1a19; }
  .title, .group, .name.ours, .value.ours { fill: #f0efec; }
  .sub, .name, .value { fill: #c3c2b7; }
  .axis { stroke: #383835; }
  .ours-fill { fill: #3987e5; }
  .ref-fill { fill: #6e6d68; }
  .l0 { fill: #184f95; } .l1 { fill: #256abf; } .l2 { fill: #3987e5; } .l3 { fill: #6da7ec; } .l4 { fill: #9ec5f4; }
}
"""

W = 760
LABEL_W = 200
VALUE_W = 120
BAR_W = W - LABEL_W - VALUE_W - 24
ROW = 26
BAR = 16
GROUP_HEAD = 30


def load(criterion: Path):
    """{group: {function: elements per second}} from the latest run."""
    groups = {}
    for bench in criterion.glob("*/*/new/benchmark.json"):
        meta = json.loads(bench.read_text())
        est = json.loads((bench.parent / "estimates.json").read_text())
        tp = meta.get("throughput") or {}
        elements = tp.get("Elements") or tp.get("Bytes")
        if not elements:
            continue
        ns = est["mean"]["point_estimate"]
        groups.setdefault(meta["group_id"], {})[meta["function_id"]] = elements / (ns * 1e-9)
    return groups


def ratios(log):
    """{(group, implementation): compression ratio} from the benches' stderr."""
    out = {}
    if log:
        pat = re.compile(r"^(\S.*?): (.+) compression ratio ([0-9.]+)$")
        for line in Path(log).read_text(errors="replace").splitlines():
            m = pat.match(line.strip())
            if m:
                out[(m[1], m[2])] = float(m[3])
    return out


def bar(x, y, w, h, cls, tip):
    """A horizontal bar: square at the baseline, 4px rounded data end."""
    r = min(4, w / 2, h / 2)
    if w <= 0.5:
        return ""
    d = (
        f"M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} "
        f"a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"
    )
    return f'<path class="{cls}" d="{d}"><title>{html.escape(tip)}</title></path>'


def text(x, y, s, cls, anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{html.escape(s)}</text>'


def svg(title, subtitle, legend, blocks):
    """`blocks`: [(heading, [(label, value, value_text, cls, ours)])], each
    block scaled to its own maximum."""
    out = []
    y = 34
    out.append(text(16, y, title, "title"))
    for line in subtitle.split("\n"):
        y += 19
        out.append(text(16, y, line, "sub"))
    y += 14
    if legend:
        x = 16
        y += 10
        for cls, label in legend:
            out.append(f'<rect class="{cls}" x="{x}" y="{y - 10}" width="12" height="12" rx="3"/>')
            out.append(text(x + 18, y, label, "sub"))
            x += 30 + 7.2 * len(label)
        y += 8
    for heading, rows in blocks:
        y += GROUP_HEAD
        out.append(text(16, y - 9, heading, "group"))
        top = y
        vmax = max(v for _, v, _, _, _ in rows) or 1
        for label, v, vtext, cls, ours in rows:
            w = BAR_W * v / vmax
            cy = y + (ROW - BAR) / 2
            out.append(text(LABEL_W - 10, y + ROW / 2 + 4, label, "name ours" if ours else "name", "end"))
            out.append(bar(LABEL_W, cy, w, BAR, cls, f"{label}: {vtext}"))
            out.append(text(LABEL_W + w + 8, y + ROW / 2 + 4, vtext, "value ours" if ours else "value"))
            y += ROW
        out.append(f'<line class="axis" x1="{LABEL_W}" y1="{top}" x2="{LABEL_W}" y2="{y}"/>')
    h = y + 20
    body = "\n".join(out)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h:.0f}" viewBox="0 0 {W} {h:.0f}" '
        f'role="img" aria-label="{html.escape(title)}">\n<style>{STYLE}</style>\n'
        f'<rect class="bg" width="{W}" height="{h:.0f}" rx="8"/>\n{body}\n</svg>\n'
    )


def heading(group):
    """'jpeg-decode/REF 8-bit q90' -> 'Decode · REF 8-bit q90'."""
    kind, _, name = group.partition("/")
    return f"{kind.split('-', 1)[1].capitalize()} · {name}"


def machine():
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return cpu


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--criterion", type=Path, default=root / "target" / "criterion")
    ap.add_argument("--out", type=Path, default=root / "benchmarks" / "results")
    ap.add_argument("--log", help="stderr of cargo bench, for compression ratios")
    ap.add_argument("--machine", default=machine())
    ap.add_argument("--level", default=os.environ.get("DICOM_JPEG_SIMD", "the best SIMD level"))
    args = ap.parse_args()

    groups = load(args.criterion)
    ratio = ratios(args.log)
    args.out.mkdir(parents=True, exist_ok=True)
    date = datetime.date.today().isoformat()
    md = [
        "# Benchmark results",
        "",
        f"Single-threaded throughput on {args.machine} ({date}); dicom-jpeg at {args.level}.",
        "Generated by `benchmarks/plot.py` from `cargo bench -p dicom-jpeg-benchmarks`;",
        "see [../README.md](../README.md) for the method and inputs.",
        "",
    ]

    for prefix, title in CODECS:
        names = sorted(
            g for g in groups if g.split("/")[0] in (f"{prefix}-decode", f"{prefix}-encode")
        )
        if not names:
            continue
        blocks = []
        md += [f"## {title}", "", f"![{title}]({prefix}.svg)", ""]
        md += ["| Benchmark | Implementation | Mpixel/s | Compression ratio |", "|---|---|---|---|"]
        for g in names:
            impls = {f: v for f, v in groups[g].items() if not LEVEL_SUFFIX.search(f)}
            if "dicom-jpeg" not in impls:
                continue
            refs = sorted((f for f in impls if f != "dicom-jpeg"), key=lambda f: -impls[f])
            rows = []
            for f in ["dicom-jpeg"] + refs:
                v = impls[f] / 1e6
                ours = f == "dicom-jpeg"
                r = ratio.get((g, f))
                label = f"{v:,.1f}" + (f" · ratio {r:.2f}" if r else "")
                rows.append((f, v, label, "ours-fill" if ours else "ref-fill", ours))
                name = "**dicom-jpeg**" if ours else f
                md.append(f"| {heading(g)} | {name} | {v:,.1f} | {f'{r:.2f}' if r else ''} |")
            blocks.append((heading(g), rows))
        md.append("")
        sub = "Mpixel/s, higher is better · each benchmark scaled to its fastest"
        if any(k[0] in names for k in ratio):
            sub += " · encoders with compression ratio"
        sub += f"\nSingle thread · {args.machine}"
        legend = [("ours-fill", "dicom-jpeg"), ("ref-fill", "reference implementations")]
        (args.out / f"{prefix}.svg").write_text(svg(title, sub, legend, blocks))

    kernels = sorted(g for g in groups if g.startswith("kernel/"))
    if kernels:
        blocks = []
        md += ["## SIMD kernels", "", "![SIMD kernels](kernels.svg)", ""]
        md += ["| Kernel | Level | Gsamples/s | Speed-up |", "|---|---|---|---|"]
        for g in kernels:
            levels = [l for l in LEVELS if l in groups[g]]
            base = groups[g].get("scalar")
            rows = []
            for i, l in enumerate(levels):
                v = groups[g][l] / 1e9
                speed = f"{groups[g][l] / base:.1f}×" if base else ""
                rows.append((l, v, f"{v:,.2f} G/s  {speed}", f"l{i}", False))
                md.append(f"| {g.removeprefix('kernel/')} | {l} | {v:,.2f} | {speed} |")
            blocks.append((g.removeprefix("kernel/"), rows))
        md.append("")
        sub = f"Samples/s and speed-up over scalar · 64 K samples in cache\nSingle thread · {args.machine}"
        (args.out / "kernels.svg").write_text(svg("SIMD kernels", sub, None, blocks))

    (args.out / "README.md").write_text("\n".join(md))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
