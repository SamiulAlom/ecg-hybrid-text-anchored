# Prompt v2: methodology figure (final hybrid model)

Create a publication-quality methodology figure for a Springer Q1 medical-AI journal.
Use the layout of the previous attempt with four panels in one row plus an evaluation strip, but follow every rule below exactly.

## Canvas and style (strict)
- Landscape, 2400 × 1250 px (or vector), white background.
- Keep a 40 px empty margin on all four sides. **Nothing may touch or cross the edge.** Every panel title, box and text line must be fully visible; no cropped letters.
- Flat vector style, thin rounded boxes, no shadows, no 3D, no gradients, no clip-art people.
- One sans-serif font (Arial/Helvetica) throughout. Panel titles 22 px bold; box titles 16 px bold; body text at least 13 px; never smaller than 12 px. Keep all text horizontal (no rotated text).
- Colour-blind-safe palette:
  - blue #0072B2: ECG/encoder and linear head
  - orange #E69F00: text path
  - green #009E73: objective
  - vermillion #D55E00: "?" unknown entries
  - grey #888888: frozen, held-out and EMA elements
- Arrows: thin dark grey, orthogonal (horizontal or vertical segments only), with clear arrowheads. No arrow crosses text.
- Spell every word exactly as written below. No extra words, no duplicated lines, no typos. Use proper symbols: θ, θ′, μ, σ, ℝ, →, ≥, ½.

## Panel (a): Training databases (left, about 22% of width)
Title: **(a) Training databases**

- Four stacked database cylinders, each labelled on ONE line:
  - "CPSC (China)"
  - "PTB-XL (Germany)"
  - "Georgia (USA)"
  - "Shaoxing–Ningbo (China)"
- One extra grey dashed cylinder to the right, labelled "Held-out target database (never seen in training)".
- Next to the cylinders, a small grid with:
  - 3 rows labelled R1, R2, R3
  - 5 columns with HORIZONTAL headers: AF, RBBB, IAVB, LAD, Brady
  - cells containing "1", "0" or a vermillion "?" (at least one "?" per row)
- Under the grid, a legend line: vermillion "?" = "diagnosis never coded by that record's source database".
- Two lines with icons:
  - red ✗ "Naive pooling: ? → 0 (false negative; learns the source)"
  - green ✓ "Ours: ? is ignored (masked)"
- At the bottom of the panel, ONE summary line (write it only once): "4 public databases · 87,661 12-lead ECGs · 43 diagnoses".

## Panel (b): ECG encoder (about 18% of width)
Title: **(b) Shared ECG encoder**

1. A small 12-lead ECG strip icon with the caption "12-lead ECG: band-pass 0.5–45 Hz, 100 Hz, random 10 s crop (1000 samples), ±5 mV".
2. ↓ Box "Augmentation: amplitude scaling · Gaussian noise · lead dropout (p = 0.1)".
3. ↓ Blue box "1D SE-ResNet encoder", drawn as four blocks of increasing height labelled "64 → 128 → 192 → 256 channels".
4. ↓ A row of small blue squares labelled "ECG tokens H ∈ ℝ^(32×256)".
5. From the tokens, two arrows go right: one to the linear head and one (dashed) to the transformer decoder.

## Panel (c): Hybrid head (largest panel, about 38% of width)
Title: **(c) Hybrid head (two heads on the same tokens)**

### Top row, blue box "Linear head (per-diagnosis weights)"
- Contents: "mean-pool → FC 256 → 43 → linear logits".
- An arrow goes from it to the Inference box.

### Bottom, orange box "Text-anchored query head (no per-diagnosis weights)"
Steps top to bottom, numbered 1–5 in small circles:
1. A document icon with two example descriptions in italic:
   - "Right bundle branch block: QRS ≥ 120 ms, rsR′ in V1–V2 …"
   - "Bradycardia: ventricular rate below 60 bpm …"
2. Grey box with a snowflake icon: "BioLORD-2023 text encoder (frozen)" → "text anchors t_c ∈ ℝ^768".
3. "Linear + LayerNorm → label queries q_c".
4. Orange box "Transformer decoder (1 layer, 4 heads): self-attention among queries + cross-attention to ECG tokens". The dashed arrow from the ECG tokens enters this box.
5. "Cosine similarity (evidence, anchor) × τ + b → text logits".
- An arrow goes from step 5 to the Inference box.

### Right column, grey-outlined box "Inference", with two sub-boxes
- Blue-outlined sub-box "Seen diagnoses (annotated in training)", formula on its own line: "p = ½ [σ(linear) + σ(text)]", with a small ⊕ where the blue and orange arrows meet.
- Orange-outlined sub-box "Unseen diagnoses (never labelled in training)", formula: "p = σ(text) only". Only the orange arrow enters it.
- Below it, a small badge: "Zero-shot: add a diagnosis by writing its description".

## Panel (d): Training objective (right, about 22% of width)
Title: **(d) Training objective**

### Green box "Annotation-aware objective"
- "Masked asymmetric loss on BOTH heads, only on entries the source annotates (a_ic = 1)"
- Formula, centred on its own line: "L = L_linear + 1.0 · L_text"
- "Implication completion: 7 clinical rules (e.g. RBBB → BBB, sinus brady → brady, IAVB → long PR)"

### Grey box "Weight averaging (EMA)", below the green box
- Formula on ONE line: "θ′ ← μ_t θ′ + (1 − μ_t) θ"
- Second line: "μ_t = min(0.999, (1 + t)/(10 + t))"
- Third line: "θ = trained weights, θ′ = averaged weights"
- Fourth line: "Averaged weights are used for validation, model selection and testing"
- Draw ONE arrow from "trained weights θ" to this box.
- Do NOT label anything "student weights" or "teacher".

## Bottom strip: evaluation (full width, light grey background)
Four equal boxes with 13 px text. All text must fit inside its box; reduce wording, never truncate.
1. **E1 Leave-one-database-out:** "Train on 3 databases, test on the 4th (4 folds × 3 seeds)"
2. **Seen vs unseen diagnoses:** "Unseen = annotated only in the held-out database"
3. **E2 Annotation removal on PTB-XL:** "0 / 25 / 50% of PTB-XL diagnoses removed from training"
4. **Baselines (all with EMA):** "Naive, masked, Li et al. hierarchy, CORAL, DANN, GroupDRO, MixStyle"

## Do not
- Add any numbers, results or AUROC values not listed above.
- Add logos, watermarks, signatures or decorative icons.
- Repeat any line of text.
- Let any element extend beyond the 40 px margin.

## Output
High-resolution PNG (at least 300 dpi at 17 cm width, about 2400 px wide), or SVG/PDF.
