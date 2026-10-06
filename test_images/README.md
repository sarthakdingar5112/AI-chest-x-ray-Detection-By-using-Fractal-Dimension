# Test Images — what each one does

Drag these into the app (**http://localhost:5173** → Analyze) to see how it behaves.
These are free, public chest X-rays. Results are the app's guesses (this is a demo, not a
real diagnosis).

## Real chest X-rays (these get ACCEPTED ✅)

| File | What you'll roughly see |
|------|--------------------------|
| `01_normal_chest.jpg` | Fairly normal — low/mild findings |
| `02_normal_chest2.png` | Fairly normal — mild findings (same image as `sample_chest_xray.png`) |
| `03_pneumonia.jpg` | Moderate findings — e.g. Effusion, Atelectasis |
| `04_tuberculosis.jpg` | **Strong findings** — Effusion ~82%, Lung Opacity ~81% (good one to demo!) |
| `07_pneumothorax.jpg` | Moderate/strong — Lung Opacity, Effusion, Mass |

For each of these you'll get: disease %s, the lung outline, the colour heatmap, and the PDF.

## Not X-rays (these get REJECTED 🚫 — that's on purpose)

| File | What happens |
|------|--------------|
| `NOT_xray_color_photo.png` | Rejected → "This looks like a color image, not a grayscale X-ray." |
| `NOT_xray_text.png` | Rejected → "No lung fields could be detected…" |

## Tips
- The heatmap **colours**: blue = smooth/normal texture, red = rough/suspicious texture
  (there's a colour bar under the picture that says this too).
- Each analysis takes ~20 seconds — that's the fractal maths working. The scanning animation
  plays while you wait.
- Want to see a big scary result? Use `04_tuberculosis.jpg`.
- Want to show the safety feature? Use one of the `NOT_xray_*` images.

> Note: AI guesses on these public images are just for demonstration. Real accuracy comes
> from proper testing on labelled data (see `RUNBOOK.md` → training).
