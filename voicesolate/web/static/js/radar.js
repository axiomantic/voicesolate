// Voicesolate Macro-Waveform & Divide-and-Conquer Radar Visualizer

export class WaveformRadar {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    this.options = Object.assign({
      onClipSelect: null,
      onSeek: null
    }, options);

    this.duration = 0;
    this.peaks = [];
    this.rms = [];
    this.clips = [];
    this.workers = {};
    this.activeClip = null;
    this.hoverClip = null;
    this.isLoading = false;
    this.loadingMessage = "Sampling media audio timeline...";

    this.initCanvas();
    this.bindEvents();
    this.startRenderLoop();
  }

  initCanvas() {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height || 220;

    this.canvas.width = this.width * this.dpr;
    this.canvas.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  resize() {
    this.initCanvas();
    this.draw();
  }

  setLoading(isLoading, message = "Sampling media audio timeline...") {
    this.isLoading = isLoading;
    this.loadingMessage = message;
    this.draw();
  }

  setData(waveformData) {
    if (!waveformData) return;
    this.isLoading = false;
    this.duration = waveformData.duration || 0;
    this.peaks = waveformData.peaks || [];
    this.rms = waveformData.rms || [];
    this.clips = waveformData.clips || [];
    this.draw();
  }

  updateWorker(workerData) {
    if (!workerData || !workerData.worker_id) return;
    this.workers[workerData.worker_id] = workerData;
    this.draw();
  }

  addClip(clipData) {
    const exists = this.clips.some(c => Math.abs(c.start_sec - clipData.start_sec) < 0.2);
    if (!exists) {
      this.clips.push(clipData);
      this.draw();
    }
  }

  formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  bindEvents() {
    window.addEventListener("resize", () => this.resize());

    this.canvas.addEventListener("mousemove", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const ratio = x / this.width;
      const hoverSec = ratio * this.duration;

      if (this.options.onHoverTime) {
        this.options.onHoverTime(hoverSec);
      }

      const found = this.clips.find(c => hoverSec >= c.start_sec && hoverSec <= c.end_sec);
      if (found !== this.hoverClip) {
        this.hoverClip = found;
        this.canvas.style.cursor = found ? "pointer" : "default";
        this.draw();
      }
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.hoverClip = null;
      this.draw();
    });

    this.canvas.addEventListener("click", (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const ratio = x / this.width;
      const clickSec = ratio * this.duration;

      const found = this.clips.find(c => clickSec >= c.start_sec && clickSec <= c.end_sec);
      this.activeClip = found || null;
      if (this.options.onClipSelect) {
        this.options.onClipSelect(this.activeClip);
      }
      this.draw();
    });
  }

  startRenderLoop() {
    const loop = () => {
      const hasActiveWorkers = Object.values(this.workers).some(w => w.state === "scanning");
      if (this.isLoading || hasActiveWorkers) {
        this.draw();
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  draw() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Loading / Sampling Animation State
    if (this.isLoading) {
      const now = Date.now() / 1000;

      // Draw background grid lines
      ctx.strokeStyle = "rgba(30, 41, 59, 0.6)";
      ctx.lineWidth = 1;
      const cols = 12;
      for (let i = 1; i < cols; i++) {
        const x = (i / cols) * w;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      // Draw simulated pulsating audio envelope bars
      const numBars = Math.floor(w / 7);
      const barW = Math.max(2, (w / numBars) - 1.5);
      for (let i = 0; i < numBars; i++) {
        const x = i * (w / numBars);
        const wave = Math.sin(i * 0.18 + now * 3.2) * 0.5 + 0.5;
        const subwave = Math.cos(i * 0.06 - now * 1.5) * 0.4 + 0.5;
        const heightFactor = Math.max(0.06, wave * subwave);
        const barH = heightFactor * (h * 0.38);

        const grad = ctx.createLinearGradient(0, midY - barH, 0, midY + barH);
        grad.addColorStop(0, "rgba(6, 182, 212, 0.25)");
        grad.addColorStop(0.5, "rgba(99, 102, 241, 0.15)");
        grad.addColorStop(1, "rgba(6, 182, 212, 0.25)");
        ctx.fillStyle = grad;
        ctx.fillRect(x, midY - barH, barW, barH * 2);
      }

      // Sweeping radar scan line & glow trail
      const sweepWidth = 140;
      const sweepX = ((now * 0.35) % 1.0) * (w + sweepWidth) - sweepWidth;
      const sweepGrad = ctx.createLinearGradient(sweepX, 0, sweepX + sweepWidth, 0);
      sweepGrad.addColorStop(0, "rgba(6, 182, 212, 0)");
      sweepGrad.addColorStop(0.8, "rgba(6, 182, 212, 0.12)");
      sweepGrad.addColorStop(1, "rgba(56, 189, 248, 0.35)");
      ctx.fillStyle = sweepGrad;
      ctx.fillRect(Math.max(0, sweepX), 0, Math.min(sweepWidth, w - sweepX), h);

      // Leading scanline
      if (sweepX + sweepWidth >= 0 && sweepX + sweepWidth <= w) {
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sweepX + sweepWidth, 0);
        ctx.lineTo(sweepX + sweepWidth, h);
        ctx.stroke();
      }

      // Sleek Glassmorphic Center Loading Pill
      const pillW = 340;
      const pillH = 42;
      const pillX = (w - pillW) / 2;
      const pillY = midY - pillH / 2;

      ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
      ctx.strokeStyle = "rgba(6, 182, 212, 0.65)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(pillX, pillY, pillW, pillH, 8);
      ctx.fill();
      ctx.stroke();

      // Pulsing indicator dot inside pill
      const pulseOpacity = (Math.sin(now * 5) + 1) / 2;
      ctx.fillStyle = `rgba(6, 182, 212, ${0.4 + 0.6 * pulseOpacity})`;
      ctx.beginPath();
      ctx.arc(pillX + 22, midY, 5, 0, Math.PI * 2);
      ctx.fill();

      // Loading text
      ctx.fillStyle = "#f8fafc";
      ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(this.loadingMessage || "Sampling media audio timeline...", pillX + 38, midY + 4);

      return;
    }

    // 1. Draw Grid Lines
    ctx.strokeStyle = "#162032";
    ctx.lineWidth = 1;
    const intervalSec = this.duration > 1200 ? 300 : 60;
    if (this.duration > 0) {
      for (let sec = intervalSec; sec < this.duration; sec += intervalSec) {
        const x = (sec / this.duration) * w;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();

        ctx.fillStyle = "#334155";
        ctx.font = "10px ui-monospace, sans-serif";
        ctx.fillText(this.formatTime(sec), x + 4, 14);
      }
    }

    // 2. Draw Macro Waveform Peaks & RMS
    if (this.peaks && this.peaks.length > 0) {
      const step = w / this.peaks.length;
      
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "rgba(6, 182, 212, 0.75)");
      grad.addColorStop(0.5, "rgba(99, 102, 241, 0.4)");
      grad.addColorStop(1, "rgba(6, 182, 212, 0.75)");
      ctx.fillStyle = grad;

      ctx.beginPath();
      for (let i = 0; i < this.peaks.length; i++) {
        const x = i * step;
        const p = Math.max(0.04, this.peaks[i]);
        const barH = (p * (h * 0.44));
        ctx.rect(x, midY - barH, Math.max(1, step - 0.5), barH * 2);
      }
      ctx.fill();

      // RMS Core
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.beginPath();
      for (let i = 0; i < this.rms.length; i++) {
        const x = i * step;
        const r = Math.max(0.02, this.rms[i]);
        const barH = (r * (h * 0.22));
        ctx.rect(x, midY - barH, Math.max(1, step - 0.5), barH * 2);
      }
      ctx.fill();
    } else {
      ctx.strokeStyle = "#1e293b";
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(w, midY);
      ctx.stroke();

      ctx.fillStyle = "#475569";
      ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Awaiting media selection to probe timeline...", w / 2, midY - 10);
      ctx.textAlign = "start";
    }

    // 3. Draw Divide-and-Conquer Search Worker Windows
    const now = Date.now() / 1000;
    Object.values(this.workers).forEach(worker => {
      if (this.duration <= 0) return;
      const x1 = (worker.chunk_start / this.duration) * w;
      const x2 = (worker.chunk_end / this.duration) * w;
      const spanW = Math.max(4, x2 - x1);

      if (worker.state === "scanning") {
        const sweepX = x1 + ((Math.sin(now * 4) + 1) / 2) * spanW;
        ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
        ctx.fillRect(x1, 0, spanW, h);

        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x1, 1, spanW, h - 2);

        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.beginPath();
        ctx.moveTo(sweepX, 0);
        ctx.lineTo(sweepX, h);
        ctx.stroke();
      } else if (worker.state === "matched") {
        ctx.fillStyle = "rgba(16, 185, 129, 0.2)";
        ctx.fillRect(x1, 0, spanW, h);
      }
    });

    // 4. Draw Dialogue Clips
    this.clips.forEach(clip => {
      if (this.duration <= 0) return;
      const x1 = (clip.start_sec / this.duration) * w;
      const x2 = (clip.end_sec / this.duration) * w;
      const clipW = Math.max(3, x2 - x1);

      const isSelected = this.activeClip && this.activeClip.id === clip.id;
      const isHovered = this.hoverClip && this.hoverClip.id === clip.id;

      if (isSelected) {
        ctx.fillStyle = "rgba(6, 182, 212, 0.4)";
        ctx.fillRect(x1, 0, clipW, h);
        ctx.strokeStyle = "#06b6d4";
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, 1, clipW, h - 2);
      } else if (isHovered) {
        ctx.fillStyle = "rgba(99, 102, 241, 0.35)";
        ctx.fillRect(x1, 0, clipW, h);
        ctx.strokeStyle = "#818cf8";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x1, 1, clipW, h - 2);
      } else {
        const confColor = (clip.confidence >= 90) ? "rgba(16, 185, 129, 0.6)" : "rgba(245, 158, 11, 0.6)";
        ctx.fillStyle = confColor;
        ctx.fillRect(x1, midY - 14, clipW, 28);
      }

      ctx.fillStyle = isSelected ? "#06b6d4" : "#10b981";
      ctx.beginPath();
      ctx.arc(x1 + clipW / 2, 8, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // 5. Draw Hover Tooltip
    if (this.hoverClip) {
      const hClip = this.hoverClip;
      const tipText = `[${hClip.character}] ${hClip.confidence}% — "${hClip.text.slice(0, 35)}..."`;
      ctx.font = "12px sans-serif";
      const metrics = ctx.measureText(tipText);
      const tipW = metrics.width + 16;
      const tipH = 24;
      let tipX = (hClip.start_sec / this.duration) * w;
      if (tipX + tipW > w - 10) tipX = w - tipW - 10;
      const tipY = h - 34;

      ctx.fillStyle = "rgba(17, 24, 39, 0.95)";
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(tipX, tipY, tipW, tipH, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.fillText(tipText, tipX + 8, tipY + 16);
    }
  }
}
