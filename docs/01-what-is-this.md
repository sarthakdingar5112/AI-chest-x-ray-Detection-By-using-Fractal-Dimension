# 1. What Is This Project?

## In one line
> A website where you upload a chest X-ray, and it tells you which lung problems might be
> there — and it uses special "fractal" maths to show you where the lung looks abnormal.

## Your project title, in plain words
The title is: **"AI-Based Detection of Abnormalities in X-Ray Images Using Fractal Dimension
Analysis."** Let's break it down:

- **X-Ray Images** → black-and-white photos of the chest (lungs, heart, ribs).
- **Abnormalities** → things that shouldn't be there — signs of illness.
- **AI-Based Detection** → a computer that learned from lots of X-rays to spot those signs.
- **Fractal Dimension Analysis** → special maths that measures how **rough** the lung
  texture is. Sick tissue is rough in a different way than healthy tissue.

So the project = **AI + fractal maths, working together, to spot lung problems in X-rays.**

## Why does this matter?
There aren't enough doctors to read every X-ray quickly. A tool that can **flag a suspicious
X-ray** and **show why** helps doctors work faster. Your project is a small demo of that idea.

## What makes yours special?
Most student projects just feed an X-ray to an AI and print "pneumonia: 87%." Yours does more:

1. **It adds fractal maths** — a clear, explainable measurement, not just a mystery AI guess.
2. **It shows a colour heatmap** — right on the X-ray, showing *where* the texture looks
   abnormal. This is the star feature and it's exactly what your title is about.
3. **It gives two opinions** — where the AI looked, and where the texture is weird. When they
   agree, that's strong evidence.

## What do you actually see when you use it?
You open the website, drag in an X-ray, wait a few seconds, and you get:

- A list of **possible problems**, each with a **% chance**.
- A big **confidence dial** for the most likely one.
- The X-ray with a **colour heatmap** (with a colour bar telling you what the colours mean).
- **Fractal numbers** (roughness, patchiness) with plain explanations.
- A **curve chart** describing the texture in detail.
- A **"Download PDF report"** button that saves a clean 2-page report.

There's also a **"Methodology" page** with a little slider toy that shows how the fractal
measurement works — great for showing your examiners you understand it.

## It only accepts real X-rays ✅
If you try to upload a selfie, a screenshot, or some text, the app **politely refuses** and
asks for a chest X-ray. It won't give fake results for a non-X-ray. (This is a nice, safe
touch that examiners like.)

## Is this a real medical device?
**No — and say so openly.** It's a **research and learning** project. Real medical tools need
years of testing and government approval. Yours shows the *idea* and the *technology* work —
which is exactly what a final-year project should do.

👉 **Next:** [`02-concepts-simple.md`](02-concepts-simple.md) — the ideas, explained simply.
