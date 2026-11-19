# GenStudio AI — Interactive Generative Studio

A fully **client-side** educational web application that demonstrates core ideas from generative modeling, class-conditional image/pattern synthesis, and autoregressive sequence generation.

Inspired by the concepts in the original GenStudio AI project (Gaussian digit synthesis, independent judge evaluation, and character-level autoregressive language modeling), this version runs entirely in the browser with no Python backend.

## Overview

GenStudio AI lets you:

- **Synthesize** synthetic 8×8 grayscale digit-like patterns by sampling from class-conditional Gaussians `P(image | digit)`.
- **Evaluate** generated samples with a simple independent template-matching “judge”.
- **Generate text** token-by-token with a miniature character-level autoregressive model (bigram / trigram) and live temperature control.
- **Visualize** next-token probability distributions and generation entropy with charts.
- **Explore** generative vs discriminative concepts and sample from 2D Gaussian mixtures.

Everything executes in the browser using HTML5 Canvas and vanilla JavaScript. No server, no API keys, no build step.

## Features

### Pattern Synthesis
- Select target digit (0–9), number of samples, noise level (σ), and random seed.
- Draws samples from `N(μ_digit, σ²I)` using hand-crafted educational templates.
- Renders each sample as a pixelated 8×8 canvas tile.
- Independent nearest-template judge reports agreement rate with the target class.

### Autoregressive Sequence Generation
- Character-level language model trained on a short educational corpus.
- Supports bigram (`n=2`) and trigram (`n=3`) context.
- Temperature scaling of the next-token distribution (low T = deterministic, high T = creative).
- Live bar chart of the current next-token probabilities.
- Entropy-over-time line chart tracking uncertainty during generation.
- Step-by-step or continuous generation modes.

### Concepts Panel
- Short explanations of generative models, discriminative models, autoregressive factorization, and the importance of modeling correlations.
- Interactive 2D canvas demo: sample points from class-conditional isotropic Gaussians and see the density ellipses.

## Tech Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| Markup       | HTML5                                           |
| Styling      | CSS3 (custom properties, modern dark theme)     |
| Logic        | Vanilla JavaScript (ES6+)                       |
| Charts       | Chart.js 4 (CDN)                                |
| Graphics     | HTML5 Canvas                                    |
| Runtime      | Any modern browser                              |

No frameworks, no bundler, no Node.js required.

## Local Preview

1. Place the three files in the same directory:
   - `index.html`
   - `style.css`
   - `main.js`
2. Open `index.html` directly in a modern browser  
   (or serve the folder with any static file server, e.g. `python -m http.server`).

That is all. The application is self-contained and ready for static hosting.

## Educational Notes

- The digit templates are simplified hand-crafted means, not trained on real MNIST/Digits data. They still illustrate the *idea* of sampling from `P(X | Y)`.
- The autoregressive model is deliberately tiny (character-level n-gram) so the transition statistics and temperature effect remain transparent.
- The “judge” is a nearest-mean classifier; it mirrors the original project’s use of an independent evaluator for synthetic samples.

Enjoy exploring generative AI concepts interactively!
