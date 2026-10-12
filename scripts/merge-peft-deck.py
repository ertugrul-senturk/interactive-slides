#!/usr/bin/env python3
"""Assemble the combined talk deck public/decks/2026-10-peft-vpt-lora.html from the VPT and LoRA decks.

    python scripts/merge-peft-deck.py            # refuses to overwrite an existing combined deck
    python scripts/merge-peft-deck.py --force    # rebuild it from the two source decks

The talk: a new title slide, the VPT paper slides, the LoRA paper slides, the combined experiments
(s-exp, s-exp-setup, s-exp-results, s-exp-cost, taken from the LoRA deck), the live demo with a fourth
column for VPT-deep, a new summary and the merged references.

How the two decks are fitted together:
  * One shared base (:root, type, layout, controls) from the VPT deck, plus base rules only the LoRA deck has.
  * Every slide gets data-part="vpt" or data-part="lora". Global (not #id-scoped) CSS rules after the shared
    base are scoped to their own part with [data-part="..."], because both decks define classes such as
    .copies, .big, .lc, .methodgrid or pre.code differently.
  * Slide and element ids of the LoRA part that collide with the VPT part are renamed (s-method -> s-lora-method,
    cb-red -> lora-cb-red), in its HTML, CSS and JS.
  * One navigation; each part's own helpers and per-slide JS run in their own block { const ASSETS = ...; ... },
    so ASSETS, hbars(), VTAB and const RESULTS stay per part. demo/lora/fill_deck.py rewrites the one line that
    starts with "const RESULTS = " (in the LoRA block).
  * Phrases such as "last time" or "LoRA next time" are rewritten with the explicit list in WORDING below.
  * Elements with data-merged-text="..." get that text in the combined deck; <html data-merged> tells the
    experiments JS it runs in the combined deck.

It also writes scripts/pptx-prep/2026-10-peft-vpt-lora.js (the VPT prep script without its demo part, plus the
LoRA prep script's demo part), with the same --force rule. Once the combined deck is revised by hand, it is the
source of truth, so don't re-run this with --force unless the hand edits may be thrown away.
"""
import argparse
import html as htmllib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECKS = ROOT / "public" / "decks"
VPT_FILE = DECKS / "2026-10-vpt-visual-prompt-tuning.html"
LORA_FILE = DECKS / "2026-10-lora-low-rank-adaptation.html"
OUT_FILE = DECKS / "2026-10-peft-vpt-lora.html"
PREP_VPT = ROOT / "scripts" / "pptx-prep" / "2026-10-vpt-visual-prompt-tuning.js"
PREP_LORA = ROOT / "scripts" / "pptx-prep" / "2026-10-lora-low-rank-adaptation.js"
PREP_OUT = ROOT / "scripts" / "pptx-prep" / "2026-10-peft-vpt-lora.js"

VPT_SKIP = {"s-part2", "s-mine", "s-slopes", "s-three", "s-demo", "s-take", "s-refs"}
LORA_FIRST, LORA_LAST = "s-recap", "s-limits"
EXP_IDS = ["s-exp", "s-exp-setup", "s-exp-results", "s-exp-cost"]
NEW_IDS = {"s-take", "s-refs", "s-refs-2"}          # slides written by this script
# Each paper keeps its own title slide, so the two parts stay clearly separate: VPT's title opens the talk, LoRA's
# title (renamed s-lora-title) opens the second part, followed by its "From VPT to LoRA" bridge.

# Pacing for a 75-minute slot: planned minutes per slide (final ids), shown in the presenter window. About 69 minutes
# of talk, leaving time for questions. Estimates, to be adjusted after a rehearsal.
TALK_MIN = 75
PLAN = {
    "s-title": 1, "s-pretrain": 1.5, "s-storage": 1.5, "s-protocols": 1.5, "s-history": 1.5, "s-vit": 2, "s-sidebyside": 1.5,
    "s-method": 2.5, "s-why": 1.5, "s-calc": 1.5, "s-recipe": 1, "s-bench": 1, "s-main": 2, "s-data": 1.5, "s-scale": 1,
    "s-location": 1.5, "s-length": 1, "s-features": 1.5, "s-limits": 1.5, "s-cost": 1.5,
    "s-lora-title": 0.5, "s-recap": 0.5, "s-copies": 1, "s-before": 1.5, "s-lowrank": 2.5, "s-lora-method": 2, "s-tiny": 2,
    "s-where": 1.5, "s-vs": 1.5, "s-lora-calc": 1, "s-width": 1, "s-code": 1.5, "s-lang": 1.5, "s-sweep": 1, "s-which": 1,
    "s-rank": 1, "s-subspace": 1.5, "s-part": 0.5, "s-vtab": 2, "s-pertask": 1, "s-medical": 1.5, "s-lora-limits": 1.5,
    "s-exp": 0.5, "s-exp-setup": 1.5, "s-exp-results": 2.5, "s-exp-cost": 2, "s-demo": 4,
    "s-take": 1.5, "s-refs": 0, "s-refs-2": 0,
}
BASE_CSS_SECTIONS = {"type", "layout helpers", "my experiments section", "controls"}

WARNINGS = []


def warn(msg):
    print("WARNING:", msg)
    WARNINGS.append(msg)


def note(msg):
    print("  " + msg)


# ---------------------------------------------------------------- wording
# (part, slide id in the source deck, old text, new text). Applied to the slide's HTML, notes included.
WORDING = [
    ("vpt", "s-title", "Opening paper of a two-part series on parameter-efficient fine-tuning (LoRA is next).",
     "Two papers on parameter-efficient fine-tuning today: first VPT, then LoRA, then my own runs of both."),
    ("vpt", "s-title", "at the end I will run a model I trained with it, live, in this browser. Thirty minutes; please interrupt.",
     "at the end I will run the models I trained, live, in this browser. About seventy minutes in all; please interrupt."),
    ("lora", "s-title", "Second paper of the parameter-efficient fine-tuning series, after VPT.",
     "Second paper: VPT is done, now LoRA."),
    ("lora", "s-title", "on the same benchmark and backbone as VPT last time, and on medical images. About thirty minutes; please interrupt.",
     "on the same benchmark and backbone as the VPT results we just saw, and on medical images. After that come my own experiments with both methods."),
    ("vpt", "s-history", "LoRA, the next talk, also dates from 2021",
     "LoRA, the second paper in this talk, also dates from 2021"),
    ("vpt", "s-cost", "LoRA, next time, merges its update into the weights",
     "LoRA, the second paper, merges its update into the weights"),
    ("lora", "s-recap", '<div class="rl">VPT<span>last time</span></div>',
     '<div class="rl">VPT<span>Jia et al. 2022</span></div>'),
    ("lora", "s-recap", '<div class="rl">LoRA<span>today</span></div>',
     '<div class="rl">LoRA<span>Hu et al. 2022</span></div>'),
    ("lora", "s-recap", "A one-slide bridge from last time, for anyone who missed it.",
     "A one-slide bridge from VPT to the second paper."),
    ("lora", "s-copies", "Then switch to ViT-B, the model we used last time and will use again:",
     "Then switch to ViT-B, the model from the VPT part and from my experiments:"),
    ("lora", "s-calc", "VPT-deep as last time.", "VPT-deep as before."),
    ("lora", "s-calc", "with the same accounting as last time.", "with the same accounting as for VPT."),
    ("lora", "s-part", "so the numbers sit next to last time's.", "so the numbers sit next to the VPT results."),
    ("lora", "s-part", "the same groups as last time.", "the same groups as in the VPT part."),
    ("lora", "s-vtab", "Same backbone and benchmark as the VPT talk:", "Same backbone and benchmark as the VPT paper:"),
]
# phrases that are right in the combined talk too (the MAE paper was an earlier talk of the reading group)
ALLOWED = ["MAE paper from last time"]
LEFTOVER = re.compile(r"last time|next time|VPT talk|the next talk|LoRA is next|LoRA next|for next time|"
                      r"\btoday\b|series on|two-part series", re.I)


# ---------------------------------------------------------------- small HTML helpers
TAG = re.compile(r"<!--.*?-->|<(/?)([a-zA-Z][\w-]*)((?:\s+[^\s=>/]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?)*)\s*(/?)>", re.S)
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


def element_span(s, start):
    """(end of the start tag, start of the end tag, end of the element) for the element whose start tag is at `start`."""
    depth, open_end = 0, None
    for m in TAG.finditer(s, start):
        if m.group(0).startswith("<!--"):
            continue
        close, name, _, selfclose = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        if open_end is None:
            open_end = m.end()
            if name in VOID or selfclose:
                return open_end, open_end, open_end
            depth = 1
            continue
        if name in VOID or selfclose:
            continue
        depth += -1 if close else 1
        if depth == 0:
            return open_end, m.start(), m.end()
    raise ValueError("unclosed element at %d" % start)


def ids_in(s):
    return set(re.findall(r'\bid="([^"]+)"', s))


def js_ids_created(code):
    return set(re.findall(r"""\bid=\\?["']([\w-]+)""", code)) | set(re.findall(r"""\bid:\s*['"]([\w-]+)""", code))


def js_refs(code):
    r = set(re.findall(r"""\$\$?\(\s*[`'"]#([\w-]+)""", code))
    r |= set(re.findall(r"""getElementById\(\s*[`'"]([\w-]+)""", code))
    r |= set(re.findall(r"""hooks\[\s*['"]([\w-]+)""", code))
    return r


# ---------------------------------------------------------------- CSS walker
def css_items(text):
    """Top-level items of a CSS text: ('raw', text) for comments and whitespace, ('rule', selector, body),
    ('media', prelude, inner) and ('keyframes', name, full text)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            j = text.index("*/", i) + 2
            out.append(("raw", text[i:j])); i = j; continue
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            out.append(("raw", text[i:j])); i = j; continue
        b = text.index("{", i)
        prelude = text[i:b].strip()
        depth, j = 1, b + 1
        while depth:
            if text.startswith("/*", j):
                j = text.index("*/", j) + 2; continue
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        inner = text[b + 1:j - 1]
        if prelude.startswith("@media"):
            out.append(("media", prelude, inner))
        elif "keyframes" in prelude:
            out.append(("keyframes", prelude.split()[-1], text[i:j]))
        else:
            out.append(("rule", prelude, inner))
        i = j
    return out


def split_selectors(sel):
    parts, depth, cur = [], 0, ""
    for ch in sel:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip()); cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def css_sections(css):
    """[(name, text)]: the text before the first section header is named '' (the :root and global base)."""
    heads = list(re.finditer(r"^/\* -{10} (.+?) -{10} \*/[^\n]*\n", css, re.M))
    secs = [("", css[:heads[0].start()] if heads else css)]
    for k, h in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(css)
        secs.append((h.group(1).strip(), css[h.start():end]))
    return secs


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"/\*.*?\*/", "", s, flags=re.S)).strip().rstrip(";")


# ---------------------------------------------------------------- one source deck
class Deck:
    def __init__(self, key, path):
        self.key, self.path = key, path
        self.text = path.read_text(encoding="utf8")
        t = self.text
        self.css = t[t.index("<style>") + 7:t.index("</style>")]
        stage = t[t.index('<div id="stage">'):t.rindex("<script>")]
        self.slides = {}           # id -> full HTML, including the comment line in front of it
        self.order = []
        for m in re.finditer(r'(?:<!-- =+[^\n]*-->\s*\n)?<section\b[^>]*\bid="([^"]+)"', stage):
            sid = m.group(1)
            sec_start = stage.index("<section", m.start())
            _, _, end = element_span(stage, sec_start)
            self.slides[sid] = stage[m.start():end]
            self.order.append(sid)
        script = t[t.rindex("<script>") + 8:t.rindex("</script>")]
        body = script[script.index("'use strict';") + len("'use strict';"):script.rindex("})();")]
        nav = body.index("/* ---------------- navigation ---------------- */")
        self.prefix = body[:nav].strip("\n")
        heads = list(re.finditer(r"^/\* -{16} (.+?) -{16} \*/[^\n]*\n", body, re.M))
        heads = [h for h in heads if h.start() >= nav]
        self.nav = body[heads[0].start():heads[1].start()].rstrip() + "\n"
        self.sections = []          # (name, code)
        for k, h in enumerate(heads[1:], 1):
            end = heads[k + 1].start() if k + 1 < len(heads) else len(body)
            self.sections.append((h.group(1).strip(), body[h.start():end].rstrip() + "\n"))
        assert self.sections[-1][0] == "start", "last JS section of %s is not 'start'" % path.name
        self.start = self.sections.pop()[1]


# ---------------------------------------------------------------- new slides
TAKE_ROWS = [
    ('<div class="tk-num">20<small>of 24</small></div>',
     "tasks where VPT-deep beat full fine-tuning, training 0.6% of a ViT-B pretrained with labels."),
    ('<div class="tk-num">99.4<small class="pc">%</small></div>',
     "less storage per task than a fully fine-tuned copy: 2.2 MB for VPT instead of 343 MB, and 1.5 MB for LoRA."),
    ('<div class="tk-num">1.00<small class="pc">×</small></div>',
     "the test-time cost of the original model with LoRA, because its change is added into the weights. VPT's prompts cost 1.27×."),
    ('<div class="tk-num">74.6</div>',
     "mean VTAB-1k accuracy of LoRA on ViT-B/16, against 72.0 for VPT-deep and 69.0 for full fine-tuning."),
]
TAKE_STATIC_MINE = ('<div class="tk-num">3<small>datasets</small></div>',
                    "of medical images on which I trained all four methods from the same ViT-B; the measured results are on the experiment slides.")
TAKE_SRC = ("Jia et al. 2022, Table 1 (24 tasks, ViT-B/16) and Table 12; 0.6% is VPT-deep with 50 prompts per layer and a 102-class head, "
            "LoRA is rank 8 on the query and value weights with the same head. Zhang et al. 2022, Table 1, mean of the three VTAB-1k groups. "
            "Medical datasets: my runs, balanced accuracy on the test set.")
TAKE_NOTES = ("The slide to leave up while summing up. First, VPT: on a ViT-B pretrained with labels, VPT-deep beat full fine-tuning on 20 of the 24 tasks "
              "while training 0.6% of the weights. Second, what both methods buy: a small file per task instead of a whole copy, 99.4% less storage for VPT, "
              "and LoRA's file is even smaller. Third, where they differ: VPT's prompts are inputs, so every image pays for them, 1.27 times the compute with 50 prompts; "
              "LoRA's change is added into the weights, so the model runs exactly as fast as before. Fourth, accuracy on the same benchmark: LoRA 74.6 on VTAB-1k, "
              "VPT-deep 72.0, full fine-tuning 69.0, though neither cheap method wins every task. Last, my own runs on the three medical datasets, "
              "with all four methods side by side.")

REFS_1 = """<!-- ============================================================ references (written by merge-peft-deck.py) -->
<section class="slide" id="s-refs" data-title="References">
  <h2>References</h2>
  <div class="refs">
    <div>
      <h3>The two papers</h3>
      <ol>
        <li><span class="au">Jia, Tang, Chen, Cardie, Belongie, Hariharan &amp; Lim 2022.</span> <i>Visual prompt tuning.</i> ECCV. arXiv:2203.12119.</li>
        <li><span class="au">Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang &amp; Chen 2022.</span> <i>LoRA: low-rank adaptation of large language models.</i> ICLR. arXiv:2106.09685.</li>
      </ol>
      <h3>Prompting before VPT</h3>
      <ol>
        <li><span class="au">Lester, Al-Rfou &amp; Constant 2021.</span> <i>The power of scale for parameter-efficient prompt tuning.</i> EMNLP.</li>
        <li><span class="au">Li &amp; Liang 2021.</span> <i>Prefix-tuning: optimizing continuous prompts for generation.</i> ACL.</li>
        <li><span class="au">Zhou, Yang, Loy &amp; Liu 2022.</span> <i>Learning to prompt for vision-language models</i> (CoOp). IJCV. arXiv:2109.01134.</li>
        <li><span class="au">Bahng, Jahanian, Sankaranarayanan &amp; Isola 2022.</span> <i>Exploring visual prompts for adapting large-scale models.</i> arXiv:2203.17274.</li>
      </ol>
      <h3>Ideas LoRA builds on</h3>
      <ol>
        <li><span class="au">Li, Farkhoor, Liu &amp; Yosinski 2018.</span> <i>Measuring the intrinsic dimension of objective landscapes.</i> ICLR.</li>
        <li><span class="au">Aghajanyan, Zettlemoyer &amp; Gupta 2021.</span> <i>Intrinsic dimensionality explains the effectiveness of language model fine-tuning.</i> ACL.</li>
      </ol>
    </div>
    <div>
      <h3>Other parameter-efficient methods</h3>
      <ol>
        <li><span class="au">Houlsby et al. 2019.</span> <i>Parameter-efficient transfer learning for NLP</i> (adapter). ICML.</li>
        <li><span class="au">Lin, Madotto &amp; Fung 2020.</span> <i>Exploring versatile generative language model via parameter-efficient transfer learning.</i> Findings of EMNLP.</li>
        <li><span class="au">Rücklé et al. 2021.</span> <i>AdapterDrop: on the efficiency of adapters in Transformers.</i> EMNLP.</li>
        <li><span class="au">Ben Zaken, Goldberg &amp; Ravfogel 2022.</span> <i>BitFit: simple parameter-efficient fine-tuning for Transformer-based masked language-models</i> (Bias). ACL.</li>
        <li><span class="au">Zhang, Sax, Zamir, Guibas &amp; Malik 2020.</span> <i>Side-tuning: a baseline for network adaptation via additive side networks</i> (Sidetune). ECCV.</li>
      </ol>
      <h3>LoRA on images</h3>
      <ol>
        <li><span class="au">Zhang, Zhou &amp; Liu 2022.</span> <i>Neural prompt search</i> (NOAH). arXiv:2206.04673.</li>
        <li><span class="au">Zhu et al. 2024.</span> <i>MeLo: low-rank adaptation is better than fine-tuning for medical image diagnosis.</i> ISBI. arXiv:2311.08236.</li>
      </ol>
    </div>
  </div>
  <aside class="notes">Leave this up for questions. The references cover both papers and are split over two slides. This one has the two papers, the prompting work VPT builds on, the intrinsic-dimension papers behind LoRA's low-rank idea, the other cheap methods in the comparisons, and the two papers that tested LoRA on images. The next slide has the models and the datasets.</aside>
</section>
"""

REFS_2 = """<!-- ============================================================ references, continued (written by merge-peft-deck.py) -->
<section class="slide" id="s-refs-2" data-title="References, continued">
  <h2>References, continued</h2>
  <div class="refs">
    <div>
      <h3>Models</h3>
      <ol>
        <li><span class="au">Dosovitskiy et al. 2021.</span> <i>An image is worth 16×16 words: Transformers for image recognition at scale</i> (ViT). ICLR.</li>
        <li><span class="au">Liu et al. 2021.</span> <i>Swin Transformer: hierarchical vision Transformer using shifted windows.</i> ICCV.</li>
        <li><span class="au">Chen, Xie &amp; He 2021.</span> <i>An empirical study of training self-supervised vision Transformers</i> (MoCo v3). ICCV.</li>
        <li><span class="au">He et al. 2022.</span> <i>Masked autoencoders are scalable vision learners</i> (MAE). CVPR.</li>
        <li><span class="au">Zheng et al. 2021.</span> <i>Rethinking semantic segmentation from a sequence-to-sequence perspective with Transformers</i> (SETR). CVPR.</li>
        <li><span class="au">Chen et al. 2018.</span> <i>Encoder-decoder with atrous separable convolution for semantic image segmentation</i> (DeepLab v3+). ECCV.</li>
        <li><span class="au">Radford et al. 2021.</span> <i>Learning transferable visual models from natural language supervision</i> (CLIP). ICML.</li>
        <li><span class="au">Brown et al. 2020.</span> <i>Language models are few-shot learners</i> (GPT-3). NeurIPS.</li>
        <li><span class="au">Radford et al. 2019.</span> <i>Language models are unsupervised multitask learners</i> (GPT-2). OpenAI report.</li>
        <li><span class="au">Liu et al. 2019.</span> <i>RoBERTa: a robustly optimized BERT pretraining approach.</i> arXiv:1907.11692.</li>
      </ol>
    </div>
    <div>
      <h3>Datasets and benchmarks</h3>
      <ol>
        <li><span class="au">Zhai et al. 2019.</span> <i>A large-scale study of representation learning with the visual task adaptation benchmark</i> (VTAB). arXiv:1910.04867.</li>
        <li><span class="au">Parkhi, Vedaldi, Zisserman &amp; Jawahar 2012.</span> <i>Cats and dogs</i> (Oxford Pets). CVPR.</li>
        <li><span class="au">Zhou et al. 2019.</span> <i>Semantic understanding of scenes through the ADE20K dataset.</i> IJCV.</li>
        <li><span class="au">Zhong, Xiong &amp; Socher 2017.</span> <i>Seq2SQL: generating structured queries from natural language using reinforcement learning</i> (WikiSQL). arXiv:1709.00103.</li>
        <li><span class="au">Williams, Nangia &amp; Bowman 2018.</span> <i>A broad-coverage challenge corpus for sentence understanding through inference</i> (MNLI). NAACL.</li>
        <li><span class="au">Wang et al. 2019.</span> <i>GLUE: a multi-task benchmark and analysis platform for natural language understanding.</i> ICLR.</li>
        <li><span class="au">Jaeger et al. 2014; Moreira et al. 2012; Wang et al. 2017.</span> Shenzhen chest X-ray, INbreast and ChestX-ray14, used by MeLo.</li>
        <li><span class="au">Yang et al. 2023.</span> <i>MedMNIST v2: a large-scale lightweight benchmark for 2D and 3D biomedical image classification.</i> Scientific Data.</li>
        <li><span class="au">Tschandl, Rosendahl &amp; Kittler 2018.</span> <i>The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.</i> Scientific Data.</li>
      </ol>
    </div>
  </div>
  <aside class="notes">The backbones and segmentation models from the VPT results, the language models from the LoRA paper, and every dataset and benchmark behind the charts, including the medical ones from my own runs.</aside>
</section>
"""
# first author + year of every entry above, to spot references the source decks gain later
REF_KEYS = {"Jia 2022", "Hu 2022", "Lester 2021", "Li 2021", "Zhou 2021", "Zhou 2022", "Bahng 2022", "Li 2018",
            "Aghajanyan 2021", "Houlsby 2019", "Lin 2020", "Rücklé 2021", "Ben 2022", "Zhang 2020", "Zhang 2022",
            "Zhu 2024", "Dosovitskiy 2021", "Liu 2021", "Chen 2021", "He 2022", "Zheng 2021", "Chen 2018",
            "Radford 2021", "Brown 2020", "Radford 2019", "Liu 2019", "Zhai 2019", "Parkhi 2012", "Zhou 2019",
            "Zhong 2017", "Williams 2018", "Wang 2019", "Jaeger 2014", "Yang 2023", "Tschandl 2018"}

NEW_CSS = """
/* ---------- combined deck: summary, references (written by merge-peft-deck.py) ---------- */
/* one row per point: the key number on the left, one plain sentence on the right */
#s-take .tk{border-bottom:1px solid var(--rule);margin-top:4px}
#s-take .tk-row{display:grid;grid-template-columns:240px 1fr;gap:40px;align-items:center;padding:16px 0;border-top:1px solid var(--rule)}
#s-take .tk-num{font-family:var(--serif);font-weight:600;font-size:48px;line-height:1;color:var(--tuned);font-variant-numeric:lining-nums tabular-nums;white-space:nowrap}
#s-take .tk-num small{font-size:24px;font-weight:600;margin-left:8px}
#s-take .tk-num small.pc{font-size:34px;margin-left:1px}
#s-take .tk-row p{font-family:var(--serif);font-size:24px;line-height:1.33;color:var(--ink);margin:0;max-width:840px}
#s-refs .refs,#s-refs-2 .refs{display:grid;grid-template-columns:1fr 1fr;gap:0 56px;align-items:start}
#s-refs h3,#s-refs-2 h3{font-size:18px;line-height:1.3;margin:0 0 8px;padding-bottom:5px;border-bottom:1px solid var(--rule)}
#s-refs h3:not(:first-child),#s-refs-2 h3:not(:first-child){margin-top:17px}
#s-refs ol,#s-refs-2 ol{margin:0;padding:0;list-style:none}
#s-refs li,#s-refs-2 li{font-size:15.5px;line-height:1.38;margin:0 0 7px;padding-left:20px;text-indent:-20px;color:var(--ink-2)}
#s-refs li .au,#s-refs-2 li .au{color:var(--ink)}
#s-refs li i,#s-refs-2 li i{color:var(--ink)}
/* live demo with four models: VPT-deep joins between the linear probe and LoRA */
#s-demo .demo{grid-template-columns:320px 1fr;gap:26px 40px}
#s-demo .drop{width:320px;height:320px}
#s-demo #task button{padding:8px 13px}
#s-demo .res3{grid-template-columns:repeat(4,minmax(0,1fr));gap:24px}
#s-demo .mcol h3{font-size:21px}
#s-demo .mcol .meta{font-size:15.5px}
#s-demo .mcol[data-m="vpt"] h3,#s-demo .mcol[data-m="vpt"] .acc b{color:var(--c-vpt)}
#s-demo .mcol[data-m="vpt"] .row5 .tr i{background:var(--c-vpt)}
#s-demo .mcol[data-m="lora"] h3,#s-demo .mcol[data-m="lora"] .acc b{color:var(--c-lora)}
#s-demo .mcol[data-m="lora"] .row5 .tr i{background:var(--c-lora)}
"""

VPT_COLUMN = ('<div class="mcol" data-m="vpt"><div class="mh"><h3>VPT-deep</h3><div class="meta">shared frozen model + prompts</div>'
              '<div class="acc"></div></div><div class="bars"></div><div class="verdict"></div><div class="mtime"></div></div>')
DEMO_META = {"shared frozen model + head": "shared model + head", "shared frozen model + prompts": "shared model + prompts",
             "shared frozen model + pairs": "shared model + pairs", "its own retrained model": "own retrained model"}
DEMO_CAP = ("All four start from the same ViT-B/16, pretrained on ImageNet-21k. Linear probe, VPT-deep and LoRA share one frozen copy of it; "
            "full fine-tuning runs its own retrained copy.")
DEMO_NOTES = ("Every click runs the image through all four models right here in the browser, with onnxruntime-web. Linear probe, VPT-deep and LoRA all use "
              "the very same frozen copy of the base model in memory: VPT-deep feeds its 50 prompts per layer in beside the patches, LoRA feeds its rank-8 pairs "
              "in beside the query and value weights, and the linear probe uses neither. That is the unmerged form of LoRA, so one copy serves every task; "
              "merged, it would run at exactly the speed of the plain model, but it would be one model per task again. Full fine-tuning retrained every weight, "
              "so it is a separate model per task. Compare the times: VPT-deep is the slowest of the three shared models, because every layer carries 50 extra tokens. "
              "The percentage under each name is that model's balanced accuracy on the whole test set. Now switch to 1,000 training images: for linear probe, VPT-deep "
              "and LoRA only a small head and a file of a megabyte or two change, so it is instant; full fine-tuning has to download and load a second whole model, "
              "172 MB in 16-bit. The thumbnails are random test images, not picked to look good, so a wrong answer can happen; say so. Research demo, not a diagnostic tool.")


# ---------------------------------------------------------------- assembling
def apply_wording(slides, part):
    for p, sid, old, new in WORDING:
        if p != part:
            continue
        if sid not in slides:
            warn(f"wording: slide {sid} not in the {part} part, skipped: {old!r}")
            continue
        if old not in slides[sid]:
            warn(f"wording: not found on {part}:{sid}: {old!r}")
            continue
        slides[sid] = slides[sid].replace(old, new)
        note(f"wording {part}:{sid}: {old!r} -> {new!r}")


def apply_merged_text(s):
    starts = [m.start() for m in TAG.finditer(s) if m.group(3) and "data-merged-text=" in m.group(3)]
    for st in reversed(starts):
        open_end, close_start, _ = element_span(s, st)
        tag = s[st:open_end]
        m = re.search(r'\s+data-merged-text=("([^"]*)"|\'([^\']*)\')', tag)
        val = htmllib.unescape(m.group(2) if m.group(2) is not None else m.group(3))
        new_tag = tag[:m.start()] + tag[m.end():]
        s = s[:st] + new_tag + htmllib.escape(val, quote=False) + s[close_start:]
    if starts:
        note(f"data-merged-text: replaced the text of {len(starts)} element(s)")
    return s, len(starts)


def scope_css(deck, rename, dropped_ids):
    """The deck's CSS after its shared base: id rules kept (renamed), global rules scoped to the part, rules of
    slides that are not in the combined deck dropped. Returns (css text, reduced-motion selectors, keyframes)."""
    out, motion, frames = [], [], {}
    part = f'[data-part="{deck.key}"]'

    def one_rule(sel):
        keep, n_id, n_id_kept = [], 0, 0
        for p in split_selectors(sel):
            m = re.match(r"#([\w-]+)", p)
            if m:
                n_id += 1
                if m.group(1) in dropped_ids:
                    continue
                n_id_kept += 1
                keep.append(rename(p))
            elif p == ":root":
                note(f"{deck.key} CSS: kept a :root rule (custom properties; the override at the end of the style block wins)")
                keep.append(p)
            elif re.match(r"(html|body|\*|\.slide\b)", p):
                warn(f"{deck.key} CSS: global selector left unscoped: {p}")
                keep.append(p)
            else:
                keep.append(f"{part} {rename(p)}")
        return keep, n_id, n_id_kept

    for name, text in css_sections(deck.css):
        if name == "" or name in BASE_CSS_SECTIONS:
            continue
        items, sec_out, ids_total, ids_kept = css_items(text), [], 0, 0
        for it in items:
            if it[0] == "raw":
                sec_out.append(it[1])
            elif it[0] == "keyframes":
                frames.setdefault(it[1], []).append(it[2])
            elif it[0] == "media":
                if "prefers-reduced-motion" in it[1]:
                    for r in css_items(it[2]):
                        if r[0] == "rule":
                            motion += split_selectors(r[1])
                    continue
                inner = []
                for r in css_items(it[2]):
                    if r[0] == "rule":
                        k, a, b = one_rule(r[1]); ids_total += a; ids_kept += b
                        if k:
                            inner.append(",".join(k) + "{" + r[2] + "}")
                if inner:
                    sec_out.append(it[1] + "{" + "".join(inner) + "}")
            else:
                k, a, b = one_rule(it[1]); ids_total += a; ids_kept += b
                if k:
                    sec_out.append(",".join(k) + "{" + it[2] + "}")
        if ids_total and not ids_kept:
            note(f"{deck.key} CSS: dropped section '{name}' (its slides are not in the combined deck)")
            continue
        out.append(re.sub(r"\n{3,}", "\n\n", "".join(sec_out)).strip("\n") + "\n")
    return "\n".join(out), motion, frames


def base_css(vpt, lora):
    def base_text(deck):
        return "".join(t for n, t in css_sections(deck.css) if n == "" or n in BASE_CSS_SECTIONS)

    def rules(text):
        d = {}
        for it in css_items(text):
            if it[0] == "rule":
                d[norm(it[1])] = norm(it[2])
        return d
    vt, lt = base_text(vpt), base_text(lora)
    vr, lr = rules(vt), rules(lt)
    extra = []
    for sel, body in lr.items():
        if sel not in vr:
            extra.append(f"{sel}{{{body}}}")
            note(f"base CSS: only in the LoRA deck, added: {sel}")
        elif vr[sel] != body:
            if sel == ":root":
                vv = dict(x.split(":", 1) for x in vr[sel].split(";") if ":" in x)
                lv = dict(x.split(":", 1) for x in body.split(";") if ":" in x)
                diff = {k: (vv.get(k), lv.get(k)) for k in set(vv) | set(lv) if vv.get(k) != lv.get(k)}
                if diff:
                    warn(f"base CSS: :root differs, VPT values kept: {diff}")
                    for k, (a, b) in diff.items():
                        if a is None:
                            extra.append(f":root{{{k}:{b}}}")
                else:
                    note("base CSS: :root identical apart from comments")
            else:
                warn(f"base CSS: '{sel}' differs between the decks, VPT kept (LoRA: {body})")
    for sel in vr:
        if sel not in lr:
            note(f"base CSS: only in the VPT deck: {sel}")
    vt = vt.replace("/* ---------- my experiments section ---------- */\n", "")
    return vt.rstrip() + ("\n\n/* ---------- shared base rules only the LoRA deck had ---------- */\n" + "\n".join(extra) + "\n" if extra else "\n")


def mark_part(slide_html, key):
    return re.sub(r'<section class="slide"', f'<section class="slide" data-part="{key}"', slide_html, count=1)


def build(force):
    for f in (OUT_FILE, PREP_OUT):
        if f.exists() and not force:
            sys.exit(f"{f.relative_to(ROOT)} exists and may have been revised by hand; pass --force to rebuild it from the source decks.")
    vpt, lora = Deck("vpt", VPT_FILE), Deck("lora", LORA_FILE)
    print(f"VPT deck: {len(vpt.order)} slides; LoRA deck: {len(lora.order)} slides")

    # ---- which slides
    vpt_ids = [s for s in vpt.order if s not in VPT_SKIP]
    i0, i1 = lora.order.index(LORA_FIRST), lora.order.index(LORA_LAST)
    lora_ids = ["s-title"] + lora.order[i0:i1 + 1]          # the LoRA part opens with its own title slide
    for s in EXP_IDS:
        if s in lora.slides:
            lora_ids.append(s)
        else:
            warn(f"experiments slide {s} is not in the LoRA deck yet; skipped")
    if "s-demo" not in lora.slides:
        sys.exit("the LoRA deck has no s-demo slide")
    lora_ids.append("s-demo")
    for s in lora.order:
        if s not in lora_ids and s not in {"s-take", "s-refs"}:
            warn(f"LoRA slide {s} is not in the combined deck (not between {LORA_FIRST} and {LORA_LAST}, and not an experiments id)")

    # ---- the summary's "my runs" row: from the LoRA deck if its summary computes one from RESULTS
    mine_row, mine_ids = None, set()
    lt = lora.slides.get("s-take", "")
    m = re.search(r'<div class="tk-row">(?:(?!<div class="tk-row">).)*?\bid="tk-mine', lt, re.S)
    if m:
        st = lt.index('<div class="tk-row">', m.start())
        _, _, end = element_span(lt, st)
        mine_row = lt[st:end]
        mine_ids = ids_in(mine_row)
        note(f"summary: 'my runs' row taken from the LoRA deck (ids {sorted(mine_ids)}), filled from RESULTS by its JS")
    else:
        warn("summary: the LoRA deck has no tk-mine row; the 'my runs' row is a static sentence without a measured number")

    # ---- ids of slides that are not used (their CSS and JS are dropped)
    vpt_dropped = set()
    for s in vpt.order:
        if s not in vpt_ids:
            vpt_dropped |= {s} | ids_in(vpt.slides[s])
    lora_dropped = set()
    for s in lora.order:
        if s not in lora_ids:
            lora_dropped |= {s} | ids_in(lora.slides[s])
    lora_dropped -= mine_ids

    # ---- JS sections per part
    def keep_sections(deck, dropped):
        kept = []
        for name, code in deck.sections:
            bad = js_refs(code) & dropped
            if bad:
                note(f"{deck.key} JS: dropped section '{name}' (uses {', '.join(sorted(bad)[:4])})")
                continue
            kept.append((name, code))
        return kept
    vpt_js, lora_js = keep_sections(vpt, vpt_dropped), keep_sections(lora, lora_dropped)

    # ---- id collisions: rename in the LoRA part
    vpt_all_ids = set(vpt_ids) | NEW_IDS | mine_ids
    for s in vpt_ids:
        vpt_all_ids |= ids_in(vpt.slides[s])
    for _, code in vpt_js:
        vpt_all_ids |= js_ids_created(code)
    vpt_all_ids |= {"tk-mine-num", "tk-mine-txt"}
    lora_all_ids = set(lora_ids)
    for s in lora_ids:
        lora_all_ids |= ids_in(lora.slides[s])
    for _, code in lora_js:
        lora_all_ids |= js_ids_created(code)
    collide = sorted(lora_all_ids & vpt_all_ids)
    renames = {}
    for i in collide:
        renames[i] = ("s-lora-" + i[2:]) if i.startswith("s-") else ("lora-" + i)
    for a, b in renames.items():
        note(f"rename in the LoRA part: {a} -> {b}")
    if renames:
        alt = "|".join(re.escape(k) for k in sorted(renames, key=len, reverse=True))
        rx = re.compile(r"(?<=[#\"'`])(" + alt + r")(?![\w-])")
        rename = lambda s: rx.sub(lambda m: renames[m.group(1)], s)
    else:
        rename = lambda s: s

    # ---- slides
    vslides = {s: vpt.slides[s] for s in vpt_ids}
    lslides = {s: lora.slides[s] for s in lora_ids}
    apply_wording(vslides, "vpt")
    apply_wording(lslides, "lora")

    # live demo: four columns, four-model wording
    demo = lslides["s-demo"]
    demo_notes_merged = bool(re.search(r'<aside class="notes"[^>]*data-merged-text', demo))
    demo, _ = apply_merged_text(demo)
    if 'data-m="vpt"' in demo:
        warn("demo: the LoRA deck's demo already has a VPT column; none inserted")
    else:
        st = demo.index('<div class="mcol" data-m="linear"')
        _, _, end = element_span(demo, st)
        demo = demo[:end] + "\n        " + VPT_COLUMN + demo[end:]
    for a, b in DEMO_META.items():
        demo = demo.replace(f'<div class="meta">{a}</div>', f'<div class="meta">{b}</div>')
    demo = demo.replace("one image, three models", "one image, four models")
    cm = re.search(r'(<p class="cap" id="demo-cap"[^>]*>)(.*?)(</p>)', demo, re.S)
    if cm and re.search(r"\bthree\b", cm.group(2), re.I):
        demo = demo[:cm.start(2)] + DEMO_CAP + demo[cm.end(2):]
    if not demo_notes_merged:
        nm = re.search(r'(<aside class="notes"[^>]*>)(.*?)(</aside>)', demo, re.S)
        demo = demo[:nm.start(2)] + DEMO_NOTES + demo[nm.end(2):]
    lslides["s-demo"] = demo

    for sid in lslides:
        lslides[sid], _ = apply_merged_text(lslides[sid])
    for p, sl in (("vpt", vslides), ("lora", lslides)):
        for sid, h in sl.items():
            for mm in LEFTOVER.finditer(h):
                if any(a in h[max(0, mm.start() - 40):mm.end() + 40] for a in ALLOWED):
                    continue
                ctx = h[max(0, mm.start() - 50):mm.end() + 30].replace("\n", " ")
                warn(f"wording left on {p}:{sid}: ...{ctx}...")

    take_rows = TAKE_ROWS if mine_row else TAKE_ROWS + [TAKE_STATIC_MINE]
    rows_html = "".join(f'\n    <div class="tk-row">{n}<p>{t}</p></div>' for n, t in take_rows)
    if mine_row:
        rows_html += "\n    " + mine_row.strip()
    take = (f"""<!-- ============================================================ summary (written by merge-peft-deck.py) -->
<section class="slide" id="s-take" data-title="Summary">
  <h2>Summary</h2>
  <div class="tk">{rows_html}
  </div>
  <p class="src">{TAKE_SRC}</p>
  <aside class="notes">{TAKE_NOTES}</aside>
</section>
""")

    stage = []
    for s in vpt_ids:
        stage.append(mark_part(vslides[s], "vpt"))
    for s in lora_ids:
        stage.append(mark_part(rename(lslides[s]), "lora"))
    stage += [take, REFS_1, REFS_2]
    stage_html = "\n\n".join(x.strip("\n") for x in stage)
    stage_html, n_merged = apply_merged_text(stage_html)

    # references the source decks cite that the merged lists lack
    for d in (vpt, lora):
        for au, yr in re.findall(r'<li><span class="au">([^<]*?)(\d{4})[.;]', d.slides.get("s-refs", "")):
            first = re.split(r"[ ,&]", au.strip())[0]
            if f"{first} {yr}" not in REF_KEYS:
                warn(f"references: '{au.strip()} {yr}' from the {d.key} deck is not in the merged list")

    # ---- CSS
    css_base = base_css(vpt, lora)
    vcss, vmotion, vframes = scope_css(vpt, lambda s: s, vpt_dropped | {"s-take", "s-refs"})
    lcss, lmotion, lframes = scope_css(lora, rename, lora_dropped | {"s-take", "s-refs"})
    frames = []
    for name in dict.fromkeys(list(vframes) + list(lframes)):
        bodies = list(dict.fromkeys(norm(b) for b in vframes.get(name, []) + lframes.get(name, [])))
        if len(bodies) > 1:
            warn(f"CSS: @keyframes {name} differs between the decks; the VPT one is kept")
        frames.append((vframes.get(name) or lframes.get(name))[0])
    motion = list(dict.fromkeys(vmotion + [rename(x) for x in lmotion]))
    css = (css_base
           + "\n/* ================ part 1: Visual Prompt Tuning (global rules scoped to [data-part=\"vpt\"]) ================ */\n" + vcss
           + "\n/* ================ part 2: LoRA (global rules scoped to [data-part=\"lora\"]) ================ */\n" + lcss
           + "\n" + "\n".join(frames) + "\n" + NEW_CSS
           + "\n@media (prefers-reduced-motion:reduce){" + ",".join(motion) + "{transition:none;animation:none}}\n"
           + "/* both methods are presented here; in the LoRA deck VPT is a grey baseline */\n"
           + ":root{--c-vpt:var(--tuned);--c-lora:#B5761B}\n")

    # ---- JS
    def nav_norm(s):
        return re.sub(r"'(vpt|lora)-notes'|Notes · [^']*'", "", s)
    if nav_norm(vpt.nav) != nav_norm(lora.nav):
        vl, ll = set(vpt.nav.splitlines()), set(lora.nav.splitlines())
        nav = lora.nav if vl - ll == set() or len(ll - vl) > len(vl - ll) else vpt.nav
        warn("JS: the two navigations differ; using the " + ("LoRA" if nav is lora.nav else "VPT") + " one. Lines only in VPT: "
             + repr(sorted(vl - ll))[:300] + " only in LoRA: " + repr(sorted(ll - vl))[:300])
    else:
        nav = vpt.nav
        note("JS: the two navigations are identical (apart from the notes window name)")
    nav = re.sub(r"'(vpt|lora)-notes'", "'peft-notes'", nav)
    nav = re.sub(r"'Notes · [^']*'", "'Notes · VPT and LoRA'", nav)
    # pacing: the presenter window shows, next to the slide number, by when this slide should be finished (data-min)
    old_pos = "d.getElementById('pos').textContent = `${cur + 1} / ${slides.length}` + (steps(s) ? ` · step ${step} of ${steps(s)}` : '');"
    assert old_pos in nav, "notes window position line not found"
    nav = nav.replace(old_pos, old_pos + "\n  const by = slides.slice(0, cur + 1).reduce((a, x) => a + (+x.dataset.min || 0), 0);   // planned minutes, cumulative\n"
                      "  if (by) d.getElementById('pos').textContent += ` · plan: done by ${Math.floor(by)}:${by % 1 ? '30' : '00'} of ${TALK_MIN}:00`;")
    nav = nav.replace("function syncNotes(){", f"const TALK_MIN = {TALK_MIN};   // the slot; the plan leaves the rest for questions\nfunction syncNotes(){{", 1)

    vcode = "\n".join(c for _, c in vpt_js)
    vprefix = vpt.prefix
    if re.search(r"\bRESULTS\b", vcode):
        rl = re.search(r"^const RESULTS = .*?;$", vpt.text, re.M)
        vprefix += "\n" + rl.group(0).replace("const RESULTS = ", "const RESULTS_VPT = ", 1)
        vcode = re.sub(r"\bRESULTS\b", "RESULTS_VPT", vcode)
        note("JS: a VPT paper slide reads RESULTS; kept as RESULTS_VPT")
    vprefix = re.sub(r"^const RESULTS = .*\n?", "", vprefix, flags=re.M)
    lprefix = rename(lora.prefix)
    lcode = rename("\n".join(c for _, c in lora_js))

    js = ("(function(){\n'use strict';\n"
          "// helpers for the navigation; each part's block below has its own copies of its deck's helpers\n"
          "const $ = (s, r = document) => r.querySelector(s);\nconst $$ = (s, r = document) => [...r.querySelectorAll(s)];\n\n"
          + nav
          + "\n\n/* ================ part 1: Visual Prompt Tuning, from 2026-10-vpt-visual-prompt-tuning.html ================ */\n{\n"
          + vprefix + "\n\n" + vcode
          + "}\n\n/* ================ part 2: LoRA and the experiments, from 2026-10-lora-low-rank-adaptation.html ================ */\n{\n"
          + lprefix + "\n\n" + lcode
          + "}\n\n" + vpt.start.rstrip() + "\n})();\n")
    n_results = len(re.findall(r"^const RESULTS = ", js, re.M))
    assert n_results == 1, f"expected one 'const RESULTS = ' line, found {n_results}"
    assert js.index("const RESULTS = ") > js.index("part 2: LoRA")

    # ---- pacing: planned minutes per slide (data-min), shown in the presenter window
    missing = []
    def add_min(m):
        sid = m.group(1)
        if sid not in PLAN:
            missing.append(sid)
            return m.group(0)
        return m.group(0) + f' data-min="{PLAN[sid]:g}"'
    stage_html = re.sub(r'<section class="slide"[^>]*?\bid="([^"]+)"', add_min, stage_html)
    if missing:
        warn(f"no planned minutes for: {missing}")
    note(f"pacing: {sum(PLAN.values()):g} planned minutes of a {TALK_MIN}-minute slot")

    # ---- head and file
    head = vpt.text[:vpt.text.index("<style>")]
    head = head.replace('<html lang="en">', '<html lang="en" data-merged>', 1)
    head = re.sub(r"<title>.*?</title>", "<title>VPT and LoRA</title>", head)
    # both papers, each with its own link in the index and the viewer bar (lib/decks.ts reads every <meta name="paper">)
    vpt_paper = re.search(r'<meta name="paper" content="([^"]+)"', vpt.text).group(1)
    lora_paper = re.search(r'<meta name="paper" content="([^"]+)"', lora.text).group(1)
    head = re.sub(r'<meta name="paper" content="[^"]*">',
                  f'<meta name="paper" content="{vpt_paper}" data-label="VPT paper">\n<meta name="paper" content="{lora_paper}" data-label="LoRA paper">', head, count=1)
    head = re.sub(r'<meta name="citation" content="[^"]*">', '<meta name="citation" content="Jia et al., ECCV 2022 · Hu et al., ICLR 2022">', head)
    head = re.sub(r'<meta name="summary" content="[^"]*">',
                  '<meta name="summary" content="Two ways to adapt a frozen vision transformer to a new task, learned input vectors and a small low-rank change to the weights, with my own runs of both on three medical image sets and a live demo.">', head)
    head = head.replace("<!DOCTYPE html>\n", "<!DOCTYPE html>\n<!-- Assembled by scripts/merge-peft-deck.py from 2026-10-vpt-visual-prompt-tuning.html and "
                        "2026-10-lora-low-rank-adaptation.html. Once revised by hand, this file is the source of truth: "
                        "re-running the script needs --force and overwrites it. -->\n", 1)
    out = (head + "<style>\n" + css.strip("\n") + "\n</style>\n</head>\n<body>\n<div id=\"stage\">\n\n" + stage_html
           + "\n\n</div>\n\n<script>\n" + js + "</script>\n</body>\n</html>\n")

    # ---- checks
    all_ids = re.findall(r'\bid="([^"]+)"', stage_html)
    dup = sorted({i for i in all_ids if all_ids.count(i) > 1})
    if dup:
        warn(f"duplicate element ids in the combined deck: {dup}")
    if "data-merged-text" in stage_html:
        warn("data-merged-text attributes left in the deck")

    OUT_FILE.write_text(out, encoding="utf8")
    print(f"wrote {OUT_FILE.relative_to(ROOT)}: {len(stage_html.split('<section ')) - 1} slides")
    slides = re.findall(r'<section class="slide"[^>]*\bid="([^"]+)"[^>]*\bdata-title="([^"]*)"', stage_html)
    for k, (sid, t) in enumerate(slides, 1):
        print(f"  {k:2d}  {sid:16s} {t}")

    write_prep(renames)
    print(f"{len(WARNINGS)} warning(s)")


def write_prep(renames):
    v, l = PREP_VPT.read_text(encoding="utf8"), PREP_LORA.read_text(encoding="utf8")
    mark = "  // Live demo slide:"
    if mark not in v or mark not in l:
        warn("prep: no '// Live demo slide:' marker in a prep script; the combined prep script was not written")
        return
    v_body = v[v.index("(() => {") + len("(() => {"):v.index(mark)].rstrip() + "\n"
    l_head = l[l.index("(() => {") + len("(() => {"):l.index(mark)].strip()
    for a, b in renames.items():                         # the LoRA part's own freezes (storage slider, parameter buttons)
        l_head = re.sub(r'(?<=[#"\'`])' + re.escape(a) + r"(?![\w-])", b, l_head)
    l_demo = l[l.index(mark):l.rindex("})();")].rstrip() + "\n"
    l_demo = re.sub(r"/\* DEMO_SNAPSHOT \*/.*?/\* END_DEMO_SNAPSHOT \*/", "/* DEMO_SNAPSHOT */ null /* END_DEMO_SNAPSHOT */", l_demo, flags=re.S)
    l_demo = l_demo.replace("2026-10-lora-low-rank-adaptation.html", "2026-10-peft-vpt-lora.html")
    l_demo = re.sub(r"\bthe three models\b", "the four models", l_demo)
    for a, b in renames.items():
        l_demo = re.sub(r'(?<=[#"\'`])' + re.escape(a) + r"(?![\w-])", b, l_demo)
    out = ("// Runs inside the combined VPT + LoRA deck before it is converted to PowerPoint, where the controls cannot work.\n"
           "// Assembled by scripts/merge-peft-deck.py: the VPT prep script without its demo part, then the LoRA prep script.\n"
           "// The live demo shows a recorded run (see below).\n(() => {" + v_body + "\n  {   // the LoRA part, in its own block: its names repeat the VPT part's\n  " + l_head + "\n  }\n\n" + l_demo + "})();\n")
    # keep a demo run already recorded into the combined prep script (2026-10-vpt-snapshot-demo.mjs), so a rebuild does not lose it
    if PREP_OUT.exists():
        old_snap = re.search(r"/\* DEMO_SNAPSHOT \*/(.*?)/\* END_DEMO_SNAPSHOT \*/", PREP_OUT.read_text(encoding="utf8"), re.S)
        if old_snap and old_snap.group(1).strip() != "null":
            out = re.sub(r"/\* DEMO_SNAPSHOT \*/.*?/\* END_DEMO_SNAPSHOT \*/", lambda _: old_snap.group(0), out, flags=re.S)
            note("prep: kept the demo run already recorded in the combined prep script")
    PREP_OUT.write_text(out, encoding="utf8")
    print(f"wrote {PREP_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite the combined deck and its prep script")
    build(ap.parse_args().force)
