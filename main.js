/**
 * GenStudio AI — Client-Side Generative Studio
 * Pattern synthesis (class-conditional Gaussians) + Autoregressive character LM
 * Pure vanilla JS + Canvas + Chart.js — no backend.
 */

(function () {
  "use strict";

  // =========================================================================
  // Utilities
  // =========================================================================

  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t += 0x6d2b79f5;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rng) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  // =========================================================================
  // Digit Templates (hand-crafted 8×8 means for educational demo)
  // Values roughly in [0, 16] like sklearn digits intensity scale.
  // =========================================================================

  const DIGIT_TEMPLATES = {
    0: [
      0, 0, 6,12,12, 6, 0, 0,
      0, 8,14, 4, 4,14, 8, 0,
      4,14, 4, 0, 0, 4,14, 4,
      6,12, 0, 0, 0, 0,12, 6,
      6,12, 0, 0, 0, 0,12, 6,
      4,14, 4, 0, 0, 4,14, 4,
      0, 8,14, 4, 4,14, 8, 0,
      0, 0, 6,12,12, 6, 0, 0,
    ],
    1: [
      0, 0, 0, 8,14, 0, 0, 0,
      0, 0, 6,14,14, 0, 0, 0,
      0, 0, 0, 4,14, 0, 0, 0,
      0, 0, 0, 4,14, 0, 0, 0,
      0, 0, 0, 4,14, 0, 0, 0,
      0, 0, 0, 4,14, 0, 0, 0,
      0, 0, 4,12,14, 8, 0, 0,
      0, 0, 8,14,14,12, 0, 0,
    ],
    2: [
      0, 4,12,14,12, 4, 0, 0,
      0,12, 6, 0, 4,12, 0, 0,
      0, 0, 0, 0, 8,12, 0, 0,
      0, 0, 0, 4,14, 4, 0, 0,
      0, 0, 4,14, 4, 0, 0, 0,
      0, 2,14, 4, 0, 0, 0, 0,
      0,12, 8, 4, 6,10, 0, 0,
      0,14,14,14,14,14, 0, 0,
    ],
    3: [
      0, 4,12,14,12, 4, 0, 0,
      0,10, 4, 0, 4,12, 0, 0,
      0, 0, 0, 0, 6,12, 0, 0,
      0, 0, 4,14,14, 6, 0, 0,
      0, 0, 0, 0, 4,12, 0, 0,
      0, 0, 0, 0, 4,12, 0, 0,
      0,10, 4, 0, 6,12, 0, 0,
      0, 4,12,14,12, 4, 0, 0,
    ],
    4: [
      0, 0, 0, 6,14, 0, 0, 0,
      0, 0, 4,14,14, 0, 0, 0,
      0, 2,12, 4,14, 0, 0, 0,
      0,10, 6, 0,14, 0, 0, 0,
      4,14, 4, 4,14, 4, 0, 0,
      8,14,14,14,14,14, 0, 0,
      0, 0, 0, 4,14, 0, 0, 0,
      0, 0, 0, 4,14, 0, 0, 0,
    ],
    5: [
      0,14,14,14,14,12, 0, 0,
      0,14, 4, 0, 0, 0, 0, 0,
      0,14, 8, 6, 4, 0, 0, 0,
      0,14,14,14,12, 4, 0, 0,
      0, 0, 0, 0, 4,12, 0, 0,
      0, 0, 0, 0, 4,12, 0, 0,
      0,10, 4, 0, 6,12, 0, 0,
      0, 4,12,14,12, 4, 0, 0,
    ],
    6: [
      0, 0, 4,12,12, 4, 0, 0,
      0, 4,14, 4, 0, 0, 0, 0,
      0,10, 8, 0, 0, 0, 0, 0,
      0,14,12,14,12, 4, 0, 0,
      0,14, 4, 0, 4,12, 0, 0,
      0,12, 4, 0, 4,12, 0, 0,
      0, 6,12, 4, 6,12, 0, 0,
      0, 0, 6,12,12, 4, 0, 0,
    ],
    7: [
      0,14,14,14,14,14, 0, 0,
      0, 4, 0, 0, 4,12, 0, 0,
      0, 0, 0, 0,10, 8, 0, 0,
      0, 0, 0, 4,14, 2, 0, 0,
      0, 0, 0,12, 8, 0, 0, 0,
      0, 0, 4,14, 2, 0, 0, 0,
      0, 0,10, 8, 0, 0, 0, 0,
      0, 0,12, 4, 0, 0, 0, 0,
    ],
    8: [
      0, 4,12,14,12, 4, 0, 0,
      0,12, 4, 0, 4,12, 0, 0,
      0,10, 6, 0, 6,10, 0, 0,
      0, 4,14,14,14, 4, 0, 0,
      0,10, 6, 0, 6,10, 0, 0,
      0,12, 4, 0, 4,12, 0, 0,
      0,12, 4, 0, 4,12, 0, 0,
      0, 4,12,14,12, 4, 0, 0,
    ],
    9: [
      0, 4,12,14,12, 4, 0, 0,
      0,12, 4, 0, 4,12, 0, 0,
      0,12, 4, 0, 4,12, 0, 0,
      0, 6,12, 4, 6,14, 0, 0,
      0, 0, 0, 0, 4,12, 0, 0,
      0, 0, 0, 0, 6,10, 0, 0,
      0, 0, 0, 4,14, 4, 0, 0,
      0, 0, 6,12, 8, 0, 0, 0,
    ],
  };

  /**
   * Sample one 8×8 pattern from N(μ_digit, σ² I) and clip to [0, 16].
   */
  function sampleDigit(digit, sigma, rng) {
    const mu = DIGIT_TEMPLATES[digit];
    const out = new Float32Array(64);
    for (let i = 0; i < 64; i++) {
      out[i] = clamp(mu[i] + gaussian(rng) * sigma, 0, 16);
    }
    return out;
  }

  /**
   * Simple template-matching judge: nearest mean in L2 distance.
   * Returns predicted digit and whether it matches the target.
   */
  function judgeSample(sample) {
    let bestDigit = 0;
    let bestDist = Infinity;
    for (let d = 0; d <= 9; d++) {
      const mu = DIGIT_TEMPLATES[d];
      let dist = 0;
      for (let i = 0; i < 64; i++) {
        const diff = sample[i] - mu[i];
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestDigit = d;
      }
    }
    return bestDigit;
  }

  function renderDigitCanvas(sample, size = 96) {
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    const cell = size / 8;
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const v = sample[y * 8 + x];
        // Map 0..16 → grayscale (bone-like inverted feel)
        const g = Math.round(255 * (1 - v / 16));
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
      }
    }
    return canvas;
  }

  // =========================================================================
  // Pattern Synthesis UI
  // =========================================================================

  function initSynth() {
    const noiseSlider = document.getElementById("synth-noise");
    const noiseVal = document.getElementById("synth-noise-val");
    noiseSlider.addEventListener("input", () => {
      noiseVal.textContent = parseFloat(noiseSlider.value).toFixed(1);
    });

    document.getElementById("btn-generate-synth").addEventListener("click", generatePatterns);
    // Generate once on load for immediate visual feedback
    generatePatterns();
  }

  function generatePatterns() {
    const digit = parseInt(document.getElementById("synth-digit").value, 10);
    const n = clamp(parseInt(document.getElementById("synth-n").value, 10) || 8, 1, 16);
    const sigma = parseFloat(document.getElementById("synth-noise").value) || 1.2;
    const seed = parseInt(document.getElementById("synth-seed").value, 10) || 42;
    const rng = mulberry32(seed);

    const grid = document.getElementById("synth-grid");
    grid.innerHTML = "";

    let matches = 0;
    const predictions = [];

    for (let i = 0; i < n; i++) {
      const sample = sampleDigit(digit, sigma, rng);
      const pred = judgeSample(sample);
      if (pred === digit) matches++;
      predictions.push(pred);

      const cell = document.createElement("div");
      cell.className = "synth-cell";
      const canvas = renderDigitCanvas(sample, 96);
      cell.appendChild(canvas);

      const label = document.createElement("div");
      label.className = "synth-label " + (pred === digit ? "valid" : "invalid");
      label.textContent = pred === digit ? `✓ Gen #${i + 1}` : `✗ Saw ${pred}`;
      cell.appendChild(label);

      grid.appendChild(cell);
    }

    const agreement = ((matches / n) * 100).toFixed(0);
    document.getElementById("synth-metrics").hidden = false;
    document.getElementById("metric-agreement").textContent = agreement + "%";
    document.getElementById("metric-target").textContent = String(digit);
    document.getElementById("metric-n").textContent = String(n);
    document.getElementById("metric-insight").textContent =
      `The generative model drew ${n} new 64-dimensional patterns from P(image | digit=${digit}). ` +
      `An independent nearest-template judge recognized them as digit ${digit} with ${agreement}% agreement.`;
  }

  // =========================================================================
  // Autoregressive Character Model
  // =========================================================================

  const DEFAULT_CORPUS =
    "the generative model learns the shape of the data and then samples from it. " +
    "an autoregressive model writes one token at a time, each conditioned on the tokens before it. " +
    "a bigram model looks back only one step, so it learns common letter pairs but forgets the rest. " +
    "modern large language models scale this autoregressive principle with transformers and attention. " +
    "by sampling from the learned probability distribution, generative ai creates novel text and code. ";

  class AutoregressiveLM {
    constructor(corpus, nGram = 2, smoothing = 0.05) {
      this.nGram = nGram;
      this.smoothing = smoothing;
      this.fit(corpus);
    }

    fit(corpus) {
      this.chars = [...new Set(corpus)].sort();
      this.vocabSize = this.chars.length;
      this.stoi = Object.fromEntries(this.chars.map((c, i) => [c, i]));
      this.itos = Object.fromEntries(this.chars.map((c, i) => [i, c]));

      if (this.nGram === 2) {
        this.counts = Array.from({ length: this.vocabSize }, () =>
          new Float32Array(this.vocabSize).fill(this.smoothing)
        );
        for (let i = 0; i < corpus.length - 1; i++) {
          const a = this.stoi[corpus[i]];
          const b = this.stoi[corpus[i + 1]];
          if (a !== undefined && b !== undefined) this.counts[a][b] += 1;
        }
        this.probs = this.counts.map((row) => {
          const s = row.reduce((a, b) => a + b, 0);
          return row.map((v) => v / s);
        });
      } else {
        // Trigram: context = last two chars → distribution over next
        this.triCounts = {};
        for (let i = 0; i < corpus.length - 2; i++) {
          const ctx = corpus[i] + corpus[i + 1];
          const next = corpus[i + 2];
          if (!this.triCounts[ctx]) {
            this.triCounts[ctx] = new Float32Array(this.vocabSize).fill(this.smoothing);
          }
          const idx = this.stoi[next];
          if (idx !== undefined) this.triCounts[ctx][idx] += 1;
        }
        this.triProbs = {};
        for (const ctx of Object.keys(this.triCounts)) {
          const row = this.triCounts[ctx];
          const s = row.reduce((a, b) => a + b, 0);
          this.triProbs[ctx] = Array.from(row, (v) => v / s);
        }
      }
    }

    getDistribution(context) {
      if (this.nGram === 2) {
        const last = context.slice(-1) || " ";
        const idx = this.stoi[last];
        if (idx === undefined) {
          // Uniform fallback
          const u = 1 / this.vocabSize;
          return this.chars.map(() => u);
        }
        return this.probs[idx];
      } else {
        const ctx = context.slice(-2);
        if (this.triProbs[ctx]) return this.triProbs[ctx];
        // Fall back to unigram-ish uniform
        const u = 1 / this.vocabSize;
        return this.chars.map(() => u);
      }
    }

    sampleNext(context, temperature, rng) {
      let probs = this.getDistribution(context).slice();
      // Temperature scaling
      if (temperature !== 1) {
        const logits = probs.map((p) => Math.log(Math.max(p, 1e-12)) / temperature);
        const maxL = Math.max(...logits);
        const exps = logits.map((l) => Math.exp(l - maxL));
        const sum = exps.reduce((a, b) => a + b, 0);
        probs = exps.map((e) => e / sum);
      }
      // Sample
      const r = rng();
      let cum = 0;
      for (let i = 0; i < probs.length; i++) {
        cum += probs[i];
        if (r < cum) return { char: this.chars[i], probs, index: i };
      }
      return { char: this.chars[probs.length - 1], probs, index: probs.length - 1 };
    }
  }

  // =========================================================================
  // Autoregressive UI + Charts
  // =========================================================================

  let arModel = null;
  let arState = {
    text: "",
    running: false,
    entropyHistory: [],
  };
  let arChart = null;
  let arProgressChart = null;

  function initAR() {
    const tempSlider = document.getElementById("ar-temp");
    const tempVal = document.getElementById("ar-temp-val");
    tempSlider.addEventListener("input", () => {
      tempVal.textContent = parseFloat(tempSlider.value).toFixed(2);
    });

    document.getElementById("btn-ar-generate").addEventListener("click", () => runGeneration(false));
    document.getElementById("btn-ar-step").addEventListener("click", () => stepOnce());
    document.getElementById("btn-ar-reset").addEventListener("click", resetAR);

    document.getElementById("ar-ngram").addEventListener("change", () => {
      rebuildModel();
      resetAR();
    });

    rebuildModel();
    initCharts();
    resetAR();
  }

  function rebuildModel() {
    const n = parseInt(document.getElementById("ar-ngram").value, 10) || 2;
    arModel = new AutoregressiveLM(DEFAULT_CORPUS, n, 0.05);
  }

  function initCharts() {
    const ctx = document.getElementById("ar-chart").getContext("2d");
    arChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: [],
        datasets: [
          {
            label: "P(next)",
            data: [],
            backgroundColor: "rgba(129, 140, 248, 0.7)",
            borderColor: "rgba(129, 140, 248, 1)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => `P = ${c.raw.toFixed(4)}`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: "#94a3b8", font: { size: 10, family: "monospace" }, maxRotation: 0 },
            grid: { color: "rgba(51,65,85,0.5)" },
          },
          y: {
            beginAtZero: true,
            max: 1,
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(51,65,85,0.5)" },
          },
        },
      },
    });

    const ctx2 = document.getElementById("ar-progress-chart").getContext("2d");
    arProgressChart = new Chart(ctx2, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Entropy (bits)",
            data: [],
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.15)",
            fill: true,
            tension: 0.25,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#94a3b8", maxTicksLimit: 8 },
            grid: { color: "rgba(51,65,85,0.4)" },
          },
          y: {
            beginAtZero: true,
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(51,65,85,0.4)" },
          },
        },
      },
    });
  }

  function entropy(probs) {
    let h = 0;
    for (const p of probs) {
      if (p > 1e-12) h -= p * Math.log2(p);
    }
    return h;
  }

  function updateDistChart(probs, chosenIndex) {
    // Show top-K most probable characters for readability
    const pairs = arModel.chars.map((c, i) => ({ c, p: probs[i], i }));
    pairs.sort((a, b) => b.p - a.p);
    const top = pairs.slice(0, 16);

    arChart.data.labels = top.map((x) => (x.c === " " ? "␣" : x.c));
    arChart.data.datasets[0].data = top.map((x) => x.p);
    arChart.data.datasets[0].backgroundColor = top.map((x) =>
      x.i === chosenIndex ? "rgba(52, 211, 153, 0.85)" : "rgba(129, 140, 248, 0.65)"
    );
    arChart.update("none");
  }

  function updateProgressChart() {
    arProgressChart.data.labels = arState.entropyHistory.map((_, i) => i + 1);
    arProgressChart.data.datasets[0].data = arState.entropyHistory;
    arProgressChart.update("none");
  }

  function resetAR() {
    arState.text = document.getElementById("ar-prompt").value || "";
    arState.running = false;
    arState.entropyHistory = [];
    document.getElementById("ar-output").textContent = arState.text || "(empty — start generating)";
    document.getElementById("ar-status").textContent = "Ready";
    if (arChart) {
      arChart.data.labels = [];
      arChart.data.datasets[0].data = [];
      arChart.update("none");
    }
    if (arProgressChart) {
      arProgressChart.data.labels = [];
      arProgressChart.data.datasets[0].data = [];
      arProgressChart.update("none");
    }
  }

  function stepOnce() {
    if (!arModel) return;
    const temperature = parseFloat(document.getElementById("ar-temp").value) || 0.9;
    const rng = Math.random; // interactive step uses fresh randomness

    if (!arState.text) {
      // Bootstrap with a random character from vocab
      const idx = Math.floor(rng() * arModel.vocabSize);
      arState.text = arModel.chars[idx];
    }

    const { char, probs, index } = arModel.sampleNext(arState.text, temperature, rng);
    arState.text += char;
    arState.entropyHistory.push(+entropy(probs).toFixed(3));

    document.getElementById("ar-output").textContent = arState.text;
    document.getElementById("ar-status").textContent = `+ '${char === " " ? "␣" : char}'`;
    updateDistChart(probs, index);
    updateProgressChart();
  }

  async function runGeneration(singleStep) {
    if (arState.running) return;
    arState.running = true;

    const maxLen = clamp(parseInt(document.getElementById("ar-length").value, 10) || 80, 5, 200);
    const temperature = parseFloat(document.getElementById("ar-temp").value) || 0.9;
    const seedInput = document.getElementById("ar-prompt").value;
    // Deterministic-ish stream from a simple seed derived from prompt + temp
    let seed = 0;
    for (let i = 0; i < seedInput.length; i++) seed = (seed * 31 + seedInput.charCodeAt(i)) >>> 0;
    seed = (seed + Math.floor(temperature * 1000)) >>> 0 || 42;
    const rng = mulberry32(seed);

    if (!arState.text) {
      arState.text = seedInput || arModel.chars[Math.floor(rng() * arModel.vocabSize)];
    }

    document.getElementById("ar-status").textContent = "Generating…";
    const target = arState.text.length + (singleStep ? 1 : maxLen);

    while (arState.text.length < target && arState.running) {
      const { char, probs, index } = arModel.sampleNext(arState.text, temperature, rng);
      arState.text += char;
      arState.entropyHistory.push(+entropy(probs).toFixed(3));

      document.getElementById("ar-output").textContent = arState.text;
      updateDistChart(probs, index);
      updateProgressChart();

      // Yield so UI stays responsive
      await new Promise((r) => setTimeout(r, 28));
    }

    document.getElementById("ar-status").textContent = "Done";
    arState.running = false;
  }

  // =========================================================================
  // Concepts — 2D Gaussian Demo
  // =========================================================================

  const DEMO_MEANS = [
    { x: -1.8, y: -1.0, color: "#38bdf8" },
    { x: 1.8, y: 1.0, color: "#f43f5e" },
  ];
  const DEMO_STD = 0.75;
  let demoPoints = [];

  function initDemo() {
    document.getElementById("btn-2d-sample").addEventListener("click", sample2D);
    document.getElementById("btn-2d-clear").addEventListener("click", () => {
      demoPoints = [];
      drawDemo();
    });
    drawDemo();
  }

  function sample2D() {
    const mode = document.getElementById("demo-class").value;
    const rng = Math.random;
    const n = 40;
    for (let i = 0; i < n; i++) {
      let cls;
      if (mode === "mix") cls = rng() < 0.5 ? 0 : 1;
      else cls = parseInt(mode, 10);
      const m = DEMO_MEANS[cls];
      demoPoints.push({
        x: m.x + gaussian(rng) * DEMO_STD,
        y: m.y + gaussian(rng) * DEMO_STD,
        cls,
      });
    }
    // Keep last ~400 points
    if (demoPoints.length > 400) demoPoints = demoPoints.slice(-400);
    drawDemo();
  }

  function drawDemo() {
    const canvas = document.getElementById("demo-canvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // Background
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = "rgba(51,65,85,0.6)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = (i / 10) * W;
      const y = (i / 10) * H;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // Axes through origin
    const toCanvas = (x, y) => {
      // World: x,y ≈ [-4,4] → canvas
      const sx = ((x + 4) / 8) * W;
      const sy = ((4 - y) / 8) * H;
      return [sx, sy];
    };

    ctx.strokeStyle = "rgba(148,163,184,0.5)";
    ctx.beginPath();
    const [ox1, oy1] = toCanvas(-4, 0);
    const [ox2, oy2] = toCanvas(4, 0);
    ctx.moveTo(ox1, oy1);
    ctx.lineTo(ox2, oy2);
    const [oy3, oy4] = [toCanvas(0, -4), toCanvas(0, 4)];
    ctx.moveTo(oy3[0], oy3[1]);
    ctx.lineTo(oy4[0], oy4[1]);
    ctx.stroke();

    // Density ellipses (1σ / 2σ)
    for (const m of DEMO_MEANS) {
      for (const k of [1, 2]) {
        ctx.beginPath();
        ctx.strokeStyle = m.color + (k === 1 ? "aa" : "55");
        ctx.lineWidth = k === 1 ? 2 : 1;
        const [cx, cy] = toCanvas(m.x, m.y);
        const rx = (DEMO_STD * k / 8) * W;
        const ry = (DEMO_STD * k / 8) * H;
        ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      // Mean marker
      const [mx, my] = toCanvas(m.x, m.y);
      ctx.fillStyle = m.color;
      ctx.beginPath();
      ctx.arc(mx, my, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Sampled points
    for (const p of demoPoints) {
      const [px, py] = toCanvas(p.x, p.y);
      ctx.fillStyle = DEMO_MEANS[p.cls].color;
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Legend
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px system-ui";
    ctx.fillText("Class 0", 16, 22);
    ctx.fillStyle = DEMO_MEANS[0].color;
    ctx.beginPath();
    ctx.arc(70, 18, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#94a3b8";
    ctx.fillText("Class 1", 16, 42);
    ctx.fillStyle = DEMO_MEANS[1].color;
    ctx.beginPath();
    ctx.arc(70, 38, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  // =========================================================================
  // Tabs
  // =========================================================================

  function initTabs() {
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll(".tab").forEach((b) => {
          b.classList.toggle("active", b === btn);
          b.setAttribute("aria-selected", b === btn ? "true" : "false");
        });
        document.querySelectorAll(".panel").forEach((p) => {
          const match = p.id === "panel-" + tab;
          p.hidden = !match;
          p.classList.toggle("active", match);
        });
      });
    });
  }

  // =========================================================================
  // Boot
  // =========================================================================

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initSynth();
    initAR();
    initDemo();
  });
})();
