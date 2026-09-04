import { api } from "./api.js";
import { WaveformRadar } from "./radar.js";

class VoicesolateApp {
  constructor() {
    this.currentEpisode = null;
    this.currentCharacter = null;
    this.activeJobId = null;
    this.radar = null;
    this.systemStatus = null;
    this.engines = [];
    this.selectedClip = null;

    this.init();
  }

  async init() {
    this.setupTabs();
    this.setupRadar();
    this.setupEventListeners();

    // Connect WebSocket
    api.connectWebSocket();
    api.subscribe((event) => this.handleSocketEvent(event));

    // Load initial system and episode data
    await this.loadSystemStatus();
    await this.loadEpisodes();

    if (this.radar) {
      setTimeout(() => this.radar.resize(), 60);
    }
  }

  switchTab(targetTab) {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(b => {
      if (b.getAttribute("data-tab") === targetTab) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });

    tabPanes.forEach(p => {
      if (p.id === targetTab) {
        p.classList.add("active");
      } else {
        p.classList.remove("active");
      }
    });

    if (targetTab === "tab-radar" && this.radar) {
      setTimeout(() => this.radar.resize(), 50);
    }
  }

  setupTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        this.switchTab(targetTab);
      });
    });
  }

  setupRadar() {
    this.radar = new WaveformRadar("macroWaveformCanvas", {
      onClipSelect: (clip) => this.handleClipSelected(clip),
      onSeek: (sec) => {
        const playTimeEl = document.getElementById("radarCursorTime");
        if (playTimeEl) playTimeEl.innerText = this.formatTime(sec);
      }
    });
  }

  formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds - Math.floor(seconds)) * 100);
    return `${m}:${s < 10 ? '0' : ''}${s}.${ms < 10 ? '0' : ''}${ms}`;
  }

  async loadSystemStatus() {
    try {
      const status = await api.getSystemStatus();
      this.systemStatus = status;
      
      const accelPill = document.getElementById("systemAccelPill");
      if (accelPill) {
        accelPill.innerHTML = `⚡ <span>${status.acceleration}</span>`;
      }
    } catch (err) {
      console.warn("Could not load system status:", err);
    }
  }

  async loadEpisodes() {
    try {
      const episodes = await api.getEpisodes();
      const epSelect = document.getElementById("headerEpisodeSelect");
      if (!epSelect) return;

      epSelect.innerHTML = "";
      if (episodes.length === 0) {
        epSelect.innerHTML = "<option value=''>No episodes found</option>";
        return;
      }

      episodes.forEach((ep, idx) => {
        const opt = document.createElement("option");
        opt.value = ep.id;
        opt.innerText = `${ep.name} (${ep.clips_count} clips)`;
        if (idx === 0) opt.selected = true;
        epSelect.appendChild(opt);
      });

      epSelect.addEventListener("change", (e) => {
        this.selectEpisode(e.target.value);
      });

      if (episodes.length > 0) {
        await this.selectEpisode(episodes[0].id);
      }
    } catch (err) {
      console.error("Error loading episodes:", err);
    }
  }

  async selectEpisode(episodeId) {
    this.currentEpisode = episodeId;
    try {
      const details = await api.getEpisodeDetails(episodeId);
      
      // Update character dropdown
      const charSelect = document.getElementById("headerCharacterSelect");
      if (charSelect) {
        charSelect.innerHTML = "";
        details.characters.forEach((char, idx) => {
          const opt = document.createElement("option");
          opt.value = char.name;
          opt.innerText = `${char.name} (${char.raw_clips_count} clips)`;
          if (idx === 0) opt.selected = true;
          charSelect.appendChild(opt);
        });

        charSelect.onchange = (e) => this.selectCharacter(e.target.value);

        if (details.characters.length > 0) {
          await this.selectCharacter(details.characters[0].name);
        }
      }

      // Load waveform for this episode
      const waveformData = await api.getWaveform(episodeId);
      if (this.radar) {
        this.radar.setData(waveformData);
        document.getElementById("radarDurationText").innerText = this.formatTime(waveformData.duration);
        document.getElementById("radarClipsCount").innerText = `${waveformData.clips ? waveformData.clips.length : 0} clips mapped`;
      }
    } catch (err) {
      console.error("Error selecting episode:", err);
    }
  }

  async selectCharacter(charName) {
    this.currentCharacter = charName;
    try {
      const details = await api.getCharacterDetails(charName, this.currentEpisode);
      this.updateEngineCards(details.engines, details.dataset_stats);
      this.updateStudioInterface(details);
    } catch (err) {
      console.error("Error selecting character:", err);
    }
  }

  updateEngineCards(engines, stats) {
    const container = document.getElementById("enginesGridContainer");
    if (!container) return;

    container.innerHTML = "";
    engines.forEach(eng => {
      const card = document.createElement("div");
      card.className = "engine-card";

      const badgeClass = eng.ready ? "badge-ready" : "badge-pending";
      const badgeText = eng.ready ? "🟢 Ready to Synthesize" : (eng.dataset_ready ? "🟡 Dataset Ready (Needs Compile)" : "⚪ Pending");

      let actionBtn = "";
      if (eng.id === "piper" && !eng.ready) {
        actionBtn = `
          <button class="btn btn-secondary btn-sm" id="setupPiperBtn">
            ⚡ Setup / Compile Piper
          </button>
        `;
      }

      card.innerHTML = `
        <div>
          <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:10px;">
            <h3 style="font-size:16px; color:#fff;">${eng.name}</h3>
            <span class="badge ${badgeClass}">${badgeText}</span>
          </div>
          <p style="font-size:12px; color:var(--text-muted); font-family:var(--font-mono); margin-bottom:8px;">${eng.architecture}</p>
          <p style="font-size:13px; color:var(--text-main); line-height:1.4;">${eng.description}</p>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:16px;">
          <span style="font-size:12px; color:var(--text-dim);">${eng.type.toUpperCase()}</span>
          ${actionBtn}
        </div>
      `;
      container.appendChild(card);
    });

    const setupBtn = document.getElementById("setupPiperBtn");
    if (setupBtn) {
      setupBtn.addEventListener("click", async () => {
        try {
          setupBtn.disabled = true;
          setupBtn.innerText = "⏳ Compiling...";
          await api.installEngine("piper");
          alert("Piper configuration triggered in background!");
        } catch (e) {
          alert("Error: " + e.message);
          setupBtn.disabled = false;
        }
      });
    }

    // Update stats bar
    if (stats) {
      const statsEl = document.getElementById("datasetStatsHeader");
      if (statsEl) {
        statsEl.innerText = `${stats.clip_count || 0} clips aggregated | LJSpeech: ${stats.piper_dataset_ready ? 'Ready' : 'Pending'} | F5: ${stats.f5_ready ? 'Ready' : 'Pending'} | XTTS: ${stats.xtts_ready ? 'Ready' : 'Pending'}`;
      }
    }
  }

  updateStudioInterface(details) {
    // Engine selector
    const engineSelect = document.getElementById("studioEngineSelect");
    if (engineSelect) {
      engineSelect.innerHTML = "";
      details.engines.forEach(eng => {
        const opt = document.createElement("option");
        opt.value = eng.id;
        opt.innerText = `${eng.name} — ${eng.ready ? 'Ready' : 'Awaiting Model'}`;
        if (!eng.ready) opt.disabled = true;
        if (eng.ready && !engineSelect.value) opt.selected = true;
        engineSelect.appendChild(opt);
      });
    }

    // Quote selector
    const quoteSelect = document.getElementById("studioQuoteSelect");
    const dialogueInput = document.getElementById("studioDialogueText");
    if (quoteSelect && details.quotes) {
      quoteSelect.innerHTML = "<option value=''>-- Select a sample quote --</option>";
      details.quotes.forEach(q => {
        const opt = document.createElement("option");
        opt.value = q;
        opt.innerText = q.length > 70 ? q.slice(0, 70) + "..." : q;
        quoteSelect.appendChild(opt);
      });

      quoteSelect.onchange = (e) => {
        if (e.target.value && dialogueInput) {
          dialogueInput.value = e.target.value;
        }
      };

      if (details.quotes.length > 0 && dialogueInput && !dialogueInput.value) {
        dialogueInput.value = details.quotes[0];
      }
    }

    // Reference prompt player
    const refAudio = document.getElementById("studioRefAudio");
    const refLabel = document.getElementById("studioRefLabel");
    if (details.reference_prompts && details.reference_prompts.length > 0) {
      const ref = details.reference_prompts[0];
      if (refAudio) refAudio.src = `/api/v1/audio/stream?path=${encodeURIComponent(ref.path)}`;
      if (refLabel) refLabel.innerText = `${ref.name} (${ref.duration}s)`;
    }
  }

  handleClipSelected(clip) {
    this.selectedClip = clip;
    const inspector = document.getElementById("clipInspectorPanel");
    if (!inspector) return;

    inspector.style.display = "block";
    document.getElementById("inspectorCharName").innerText = clip.character;
    document.getElementById("inspectorConfidenceBadge").innerText = `${clip.confidence}% Match`;
    document.getElementById("inspectorTimecode").innerText = `${this.formatTime(clip.start_sec)} ➔ ${this.formatTime(clip.end_sec)} (${clip.duration}s)`;
    document.getElementById("inspectorText").innerText = `"${clip.text}"`;

    const rawAudio = document.getElementById("inspectorRawAudio");
    const enhAudio = document.getElementById("inspectorEnhAudio");

    if (rawAudio && clip.raw_file) {
      rawAudio.src = `/api/v1/audio/stream?path=${encodeURIComponent(clip.raw_file)}`;
    }
    if (enhAudio && clip.enhanced_file) {
      enhAudio.src = `/api/v1/audio/stream?path=${encodeURIComponent(clip.enhanced_file)}`;
    }
  }

  handleSocketEvent(event) {
    if (event.type === "ws_status") {
      const dot = document.getElementById("wsPulseDot");
      const label = document.getElementById("wsStatusText");
      if (dot) {
        if (event.connected) {
          dot.classList.remove("offline");
          if (label) label.innerText = "Live Radar";
        } else {
          dot.classList.add("offline");
          if (label) label.innerText = "Reconnecting...";
        }
      }
      return;
    }

    if (event.type === "job_created" || event.type === "job_update") {
      const job = event.job;
      this.updateJobBanner(job);
    }

    if (event.type === "worker_event" && this.radar) {
      this.radar.updateWorker(event.worker);
      this.addWorkerCard(event.worker);
    }

    if (event.type === "clip_discovered" && this.radar) {
      this.radar.addClip(event.clip);
    }
  }

  updateJobBanner(job) {
    const banner = document.getElementById("activeJobBanner");
    if (!banner) return;

    if (job.status === "running" || job.status === "queued") {
      banner.classList.add("active");
      document.getElementById("jobTitleText").innerText = `[${job.stage.toUpperCase()}] ${job.message}`;
      document.getElementById("jobProgressFill").style.width = `${job.progress}%`;
      document.getElementById("jobPercentText").innerText = `${Math.round(job.progress)}%`;
      this.activeJobId = job.job_id;
    } else if (job.status === "completed") {
      banner.classList.add("active");
      document.getElementById("jobTitleText").innerText = `✓ ${job.message}`;
      document.getElementById("jobProgressFill").style.width = "100%";
      document.getElementById("jobPercentText").innerText = "100%";

      if (job.job_type === "scan" && job.result && job.result.characters) {
        this.renderDiscoveredCharacters(job.result.characters, job.params ? job.params.input_path : "", job.params ? job.params.script_path : "");
        const modalStatusText = document.getElementById("modalScanStatusText");
        const pipeStatusText = document.getElementById("pipelineScanStatusText");
        const statusMsg = `✓ Found ${job.result.characters.length} characters (${job.result.duration}s audio)`;
        if (modalStatusText) modalStatusText.innerText = statusMsg;
        if (pipeStatusText) pipeStatusText.innerText = statusMsg;
      }

      setTimeout(() => {
        banner.classList.remove("active");
        this.loadEpisodes();
      }, 3500);
    } else if (job.status === "failed") {
      banner.classList.add("active");
      document.getElementById("jobTitleText").innerText = `❌ Error: ${job.error || job.message}`;
      document.getElementById("jobProgressFill").style.background = "var(--accent-rose)";
    }
  }

  addWorkerCard(worker) {
    const container = document.getElementById("radarWorkersContainer");
    if (!container) return;

    let card = document.getElementById(`worker_card_${worker.worker_id}`);
    if (!card) {
      card = document.createElement("div");
      card.id = `worker_card_${worker.worker_id}`;
      card.className = "worker-card";
      container.appendChild(card);
    }

    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-weight:600;">${worker.worker_id}</span>
        <span class="worker-badge ${worker.state}">${worker.state}</span>
      </div>
      <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-dim);">
        Range: ${this.formatTime(worker.chunk_start)} ➔ ${this.formatTime(worker.chunk_end)}
      </div>
      <div style="font-size:12px; color:var(--text-muted); text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">
        ${worker.snippet || 'Scanning audio spectrum...'}
      </div>
    `;
  }

  renderDiscoveredCharacters(characters, inputPath, scriptPath) {
    const renderTable = (tbodyId, containerId) => {
      const tbody = document.getElementById(tbodyId);
      const container = document.getElementById(containerId);
      if (!tbody || !container) return;

      tbody.innerHTML = "";
      if (!characters || characters.length === 0) {
        tbody.innerHTML = "<tr><td colspan='4' style='text-align:center; color:var(--text-dim);'>No speaking characters discovered.</td></tr>";
        container.style.display = "flex";
        return;
      }

      characters.forEach(char => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong style="color:#fff;">${char.name}</strong></td>
          <td><span class="badge badge-ready">${char.lines} lines</span></td>
          <td style="font-family:var(--font-mono);">${char.estimated_duration_min} min (${char.estimated_duration_sec}s)</td>
          <td>
            <button class="btn btn-primary btn-sm extract-char-btn" data-char="${char.name}" style="padding:4px 10px; font-size:12px;">
              ⚡ Extract &amp; Isolate
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      });

      tbody.querySelectorAll(".extract-char-btn").forEach(btn => {
        btn.addEventListener("click", async () => {
          const charName = btn.getAttribute("data-char");
          btn.disabled = true;
          btn.innerText = "⏳ Queuing...";
          try {
            await api.runPipeline({
              input_path: inputPath,
              script_path: scriptPath || null,
              characters: [charName],
              enhance: true,
              targets: ["all"]
            });
            this.switchTab("tab-radar");
          } catch (err) {
            alert("Extraction failed: " + err.message);
            btn.disabled = false;
            btn.innerText = "⚡ Extract & Isolate";
          }
        });
      });

      container.style.display = "flex";
    };

    renderTable("pipelineCharactersTbody", "pipelineDiscoveredContainer");
  }

  setupEventListeners() {
    // Preset Chips Click Handling
    document.querySelectorAll(".preset-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const p = chip.getAttribute("data-path");
        const s = chip.getAttribute("data-script");
        const c = chip.getAttribute("data-char");

        const pInput = document.getElementById("pipelineInputPath");
        const pScript = document.getElementById("pipelineScriptPath");
        const pChar = document.getElementById("pipelineCharInput");
        if (pInput && p) pInput.value = p;
        if (pScript && s) pScript.value = s;
        if (pChar && c) pChar.value = c;
      });
    });

    // Cancel Job Button
    const cancelBtn = document.getElementById("cancelJobBtn");
    if (cancelBtn) {
      cancelBtn.addEventListener("click", async () => {
        if (this.activeJobId) {
          await api.cancelJob(this.activeJobId);
        }
      });
    }

    // Scan Button on Pipeline Tab
    const scanBtn = document.getElementById("pipelineScanBtn");
    const pipeStatusText = document.getElementById("pipelineScanStatusText");
    if (scanBtn) {
      scanBtn.addEventListener("click", async () => {
        const inputPath = document.getElementById("pipelineInputPath").value.trim();
        const scriptPath = document.getElementById("pipelineScriptPath").value.trim();
        const provider = document.getElementById("pipelineProviderSelect").value;
        if (!inputPath) {
          alert("Please enter a media file path or remote SFTP URL.");
          return;
        }
        if (pipeStatusText) pipeStatusText.innerText = "⏳ Scanning media stream...";
        scanBtn.disabled = true;
        try {
          await api.scanMedia(inputPath, scriptPath || null, provider || null);
        } catch (err) {
          if (pipeStatusText) pipeStatusText.innerText = `❌ Error: ${err.message}`;
          alert("Scan failed: " + err.message);
        } finally {
          scanBtn.disabled = false;
        }
      });
    }

    // Run Full Extraction Button
    const extractBtn = document.getElementById("pipelineExtractBtn");
    if (extractBtn) {
      extractBtn.addEventListener("click", async () => {
        const inputPath = document.getElementById("pipelineInputPath").value.trim();
        const scriptPath = document.getElementById("pipelineScriptPath").value.trim();
        const charInput = document.getElementById("pipelineCharInput").value.trim();
        if (!inputPath) {
          alert("Please enter a media file path.");
          return;
        }
        if (!charInput) {
          alert("Please specify a character to extract (e.g. CLEMENS).");
          return;
        }
        try {
          await api.runPipeline({
            input_path: inputPath,
            script_path: scriptPath || null,
            characters: [charInput],
            enhance: true,
            targets: ["all"]
          });
          this.switchTab("tab-radar");
        } catch (err) {
          alert("Extraction trigger failed: " + err.message);
        }
      });
    }

    // Synthesize Button
    const synthBtn = document.getElementById("studioSynthesizeBtn");
    if (synthBtn) {
      synthBtn.addEventListener("click", async () => {
        const engine = document.getElementById("studioEngineSelect").value;
        const text = document.getElementById("studioDialogueText").value.trim();
        const speed = parseFloat(document.getElementById("studioSpeedRange").value);
        const seed = parseInt(document.getElementById("studioSeedInput").value) || 42;

        if (!text) {
          alert("Please enter dialogue text to synthesize.");
          return;
        }

        const originalBtnHtml = synthBtn.innerHTML;
        synthBtn.disabled = true;
        synthBtn.innerHTML = "⏳ Synthesizing Voice...";

        try {
          const res = await api.synthesize({
            character_name: this.currentCharacter,
            episode_name: this.currentEpisode,
            engine: engine,
            text: text,
            speed: speed,
            seed: seed
          });

          const synthAudio = document.getElementById("studioGeneratedAudio");
          if (synthAudio) {
            synthAudio.src = res.url;
            synthAudio.play().catch(() => {});
          }

          document.getElementById("studioGeneratedMeta").innerText = `✓ Generated ${res.duration}s with ${res.engine} (samplerate: ${res.samplerate}Hz)`;
        } catch (err) {
          alert("Synthesis failed: " + err.message);
        } finally {
          synthBtn.disabled = false;
          synthBtn.innerHTML = originalBtnHtml;
        }
      });
    }

    // Speed range display sync
    const speedRange = document.getElementById("studioSpeedRange");
    const speedVal = document.getElementById("studioSpeedValue");
    if (speedRange && speedVal) {
      speedRange.addEventListener("input", (e) => {
        speedVal.innerText = `${parseFloat(e.target.value).toFixed(2)}x`;
      });
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new VoicesolateApp();
});
