import { api } from "./api.js";
import { WaveformRadar } from "./radar.js?v=2.4";

class VoicesolateWizardApp {
  constructor() {
    this.currentStep = 1;
    this.completedSteps = new Set();
    
    // Core pipeline state
    this.mediaPath = "";
    this.episodeCode = "";
    this.episodeName = "";
    this.scriptLoaded = false;
    this.scriptCharacters = [];
    this.selectedCharacter = "";
    
    // Audio & Radar
    this.radar = null;
    this.waveformData = null;
    this.alignedClips = [];
    this.selectedClip = null;
    this.activeJobId = null;
    
    // Training & Engines
    this.engines = [];
    this.characterQuotes = [];
    this.activeWorkers = {};
    
    this.init();
  }

  async init() {
    this.setupStepper();
    this.setupRadar();
    this.setupStep1Events();
    this.setupStep2Events();
    this.setupStep3Events();
    this.setupStep4Events();

    // Connect real-time WebSocket
    api.connectWebSocket();
    api.subscribe((event) => this.handleSocketEvent(event));

    // Hardware status
    await this.loadSystemStatus();

    // Auto-restore previous state if outputs exist on disk
    await this.autoRestoreState();
  }

  // ----------------- WIZARD STEPPER & NAVIGATION -----------------

  setupStepper() {
    for (let i = 1; i <= 4; i++) {
      const stepEl = document.getElementById(`stepperStep${i}`);
      if (stepEl) {
        stepEl.addEventListener("click", () => {
          if (this.canAccessStep(i)) {
            this.goToStep(i);
          }
        });
      }
    }
  }

  canAccessStep(step) {
    if (step === 1) return true;
    if (this.completedSteps.has(step)) return true;
    // Can access next step if previous step is completed
    return this.completedSteps.has(step - 1);
  }

  goToStep(step) {
    this.currentStep = step;
    
    // Update stepper pills
    for (let i = 1; i <= 4; i++) {
      const stepEl = document.getElementById(`stepperStep${i}`);
      const paneEl = document.getElementById(`step-pane-${i}`);
      const statusEl = document.getElementById(`step${i}Status`);

      if (stepEl) {
        stepEl.classList.remove("active", "completed", "locked");
        if (i === this.currentStep) {
          stepEl.classList.add("active");
          if (statusEl) statusEl.innerText = "Active";
        } else if (this.completedSteps.has(i)) {
          stepEl.classList.add("completed");
          if (statusEl) statusEl.innerText = "✓ Completed";
        } else {
          stepEl.classList.add("locked");
          if (statusEl) statusEl.innerText = "Locked";
        }
      }

      if (paneEl) {
        if (i === this.currentStep) {
          paneEl.classList.add("active");
        } else {
          paneEl.classList.remove("active");
        }
      }
    }

    // Step-specific transition actions
    if (step === 1) {
      this.onEnterStep1();
    } else if (step === 2) {
      this.onEnterStep2();
    } else if (step === 3) {
      this.onEnterStep3();
    } else if (step === 4) {
      this.onEnterStep4();
    }
  }

  markStepCompleted(step) {
    this.completedSteps.add(step);
    const statusEl = document.getElementById(`step${step}Status`);
    if (statusEl && this.currentStep !== step) {
      statusEl.innerText = "✓ Completed";
    }
    const stepEl = document.getElementById(`stepperStep${step}`);
    if (stepEl && this.currentStep !== step) {
      stepEl.classList.add("completed");
      stepEl.classList.remove("locked");
    }
    // Unlock next step in stepper
    if (step < 4) {
      const nextStepEl = document.getElementById(`stepperStep${step + 1}`);
      if (nextStepEl) {
        nextStepEl.classList.remove("locked");
      }
    }
  }

  markStepIncomplete(step) {
    for (let s = step; s <= 4; s++) {
      this.completedSteps.delete(s);
      const stepEl = document.getElementById(`stepperStep${s}`);
      const statusEl = document.getElementById(`step${s}Status`);
      if (stepEl) {
        stepEl.classList.remove("completed");
        if (this.currentStep === s) {
          stepEl.classList.add("active");
          stepEl.classList.remove("locked");
          if (statusEl) statusEl.innerText = "In Progress";
        } else {
          stepEl.classList.remove("active");
          stepEl.classList.add("locked");
          if (statusEl) statusEl.innerText = "Locked";
        }
      }
    }
  }

  // ----------------- STEP 1: CHOOSE FILES & SCRIPT -----------------

  setupStep1Events() {
    const inputPath = document.getElementById("wizardInputPath");
    const episodeInput = document.getElementById("wizardEpisodeInput");
    const providerSelect = document.getElementById("wizardProviderSelect");
    const fetchScriptBtn = document.getElementById("wizardFetchScriptBtn");
    const scriptFileInput = document.getElementById("scriptFileInput");
    const browseScriptBtn = document.getElementById("browseScriptBtn");
    const dropzone = document.getElementById("scriptDropzone");
    const searchFilter = document.getElementById("characterSearchFilter");
    const nextBtn = document.getElementById("step1NextBtn");
    const resetBtn = document.getElementById("resetStep1Btn");

    // Presets
    document.querySelectorAll(".preset-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const path = chip.getAttribute("data-path");
        const script = chip.getAttribute("data-script");
        const charName = chip.getAttribute("data-char");

        if (inputPath && path) {
          inputPath.value = path;
          this.handleMediaPathChanged(path);
        }
        if (episodeInput && script) {
          episodeInput.value = script;
          this.episodeCode = script;
        }
        if (script) {
          this.fetchScript(script, "startrek", charName);
        }
      });
    });

    // Media Path Input typing
    if (inputPath) {
      inputPath.addEventListener("input", (e) => {
        this.handleMediaPathChanged(e.target.value.trim());
      });
    }

    // Episode Input typing
    if (episodeInput) {
      episodeInput.addEventListener("input", (e) => {
        this.episodeCode = e.target.value.trim();
        this.updateStep1Checklist();
      });
    }

    // Provider select toggle
    if (providerSelect) {
      providerSelect.addEventListener("change", (e) => {
        const isUpload = e.target.value === "upload";
        document.getElementById("archiveSearchSection").style.display = isUpload ? "none" : "flex";
        document.getElementById("customUploadSection").style.display = isUpload ? "flex" : "none";
      });
    }

    // Fetch Script Button
    if (fetchScriptBtn) {
      fetchScriptBtn.addEventListener("click", async () => {
        const ep = episodeInput ? episodeInput.value.trim() : "";
        const p = inputPath ? inputPath.value.trim() : "";
        await this.fetchScript(ep || p, "startrek");
      });
    }

    // File Upload Browse
    if (browseScriptBtn && scriptFileInput) {
      browseScriptBtn.addEventListener("click", () => scriptFileInput.click());
      scriptFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // Drag and Drop
    if (dropzone) {
      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
      dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });
    }

    // Character Roster Search Filter
    if (searchFilter) {
      searchFilter.addEventListener("input", (e) => {
        const query = (e.target.value || "").toLowerCase().trim();
        const tbody = document.getElementById("wizardCharactersTbody");
        if (tbody) {
          tbody.querySelectorAll("tr").forEach(tr => {
            const charName = (tr.getAttribute("data-char-name") || "").toLowerCase();
            tr.style.display = !query || charName.includes(query) ? "" : "none";
          });
        }
      });
    }

    // Step 1 Next Button
    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        this.markStepCompleted(1);
        this.goToStep(2);
      });
    }

    // Step 1 Reset Button
    if (resetBtn) {
      resetBtn.addEventListener("click", async () => {
        if (confirm("Reset Step 1 inputs?")) {
          await api.clearStep(1);
          if (inputPath) inputPath.value = "";
          if (episodeInput) episodeInput.value = "";
          this.mediaPath = "";
          this.episodeCode = "";
          this.scriptLoaded = false;
          this.scriptCharacters = [];
          this.selectedCharacter = "";
          document.getElementById("wizardCharacterContainer").style.display = "none";
          document.getElementById("wizardScriptStatusText").innerText = "";
          document.getElementById("wizardEpisodeBadge").style.display = "none";
          this.markStepIncomplete(1);
          this.updateStep1Checklist();
        }
      });
    }
  }

  handleMediaPathChanged(path) {
    this.mediaPath = path;
    const badge = document.getElementById("wizardEpisodeBadge");
    const epInput = document.getElementById("wizardEpisodeInput");

    // Auto-detect episode code via regex
    const m = path.match(/s0?(\d+)[ex]0?(\d+)/i);
    if (m) {
      const s = parseInt(m[1], 10);
      const e = parseInt(m[2], 10);
      const code = `s${s < 10 ? '0' : ''}${s}e${e < 10 ? '0' : ''}${e}`;
      this.episodeCode = code;
      if (epInput) epInput.value = code;
      if (badge) {
        badge.innerText = `✓ Detected Season ${s} Episode ${e}`;
        badge.style.display = "inline-block";
      }
      // Derive episode folder name
      const filename = path.split("/").pop().replace(/\.[^/.]+$/, "");
      this.episodeName = filename.replace(/[^a-zA-Z0-9_\-\.]/g, "_").slice(0, 60);
    }

    this.updateStep1Checklist();
  }

  async fetchScript(identifier, provider = "startrek", preselectChar = null) {
    const statusText = document.getElementById("wizardScriptStatusText");
    const fetchBtn = document.getElementById("wizardFetchScriptBtn");
    if (fetchBtn) fetchBtn.disabled = true;
    if (statusText) statusText.innerText = "⏳ Querying script archive...";

    try {
      const res = await api.detectScript(identifier);
      if (res.detected_episode && !this.episodeCode) {
        this.episodeCode = res.detected_episode;
        const epInput = document.getElementById("wizardEpisodeInput");
        if (epInput) epInput.value = res.detected_episode;
      }

      this.scriptLoaded = true;
      this.scriptCharacters = res.characters || [];
      if (statusText) {
        statusText.innerText = `✓ Loaded ${res.dialogues_count} dialogue lines across ${this.scriptCharacters.length} characters!`;
      }
      this.populateCharacterSelector(this.scriptCharacters, preselectChar);
    } catch (err) {
      if (statusText) statusText.innerText = `❌ Error: ${err.message}`;
    } finally {
      if (fetchBtn) fetchBtn.disabled = false;
      this.updateStep1Checklist();
    }
  }

  async handleFileUpload(file) {
    const statusText = document.getElementById("wizardScriptStatusText");
    if (statusText) statusText.innerText = `⏳ Uploading and parsing ${file.name}...`;

    try {
      const res = await api.uploadScript(file);
      this.scriptLoaded = true;
      this.scriptCharacters = res.characters || [];
      if (statusText) {
        statusText.innerText = `✓ Parsed ${file.name}: ${res.dialogues_count} dialogue lines across ${this.scriptCharacters.length} characters!`;
      }
      this.populateCharacterSelector(this.scriptCharacters);
    } catch (err) {
      if (statusText) statusText.innerText = `❌ Upload failed: ${err.message}`;
    } finally {
      this.updateStep1Checklist();
    }
  }

  populateCharacterSelector(characters, preselectChar = null) {
    const container = document.getElementById("wizardCharacterContainer");
    const tbody = document.getElementById("wizardCharactersTbody");
    const searchFilter = document.getElementById("characterSearchFilter");
    if (!container || !tbody) return;

    if (searchFilter) searchFilter.value = "";
    tbody.innerHTML = "";

    characters.forEach(c => {
      const formattedDuration = c.estimated_duration_formatted || this.humanizeSpeakingTime(c.estimated_duration_sec || (c.lines * 3.2));
      const tr = document.createElement("tr");
      tr.setAttribute("data-char-name", c.name);
      tr.style.transition = "background-color 0.2s ease";
      tr.innerHTML = `
        <td><strong>${this.escapeHtml(c.name)}</strong></td>
        <td><span class="badge badge-ready">${c.lines} lines</span></td>
        <td style="font-family:var(--font-mono); font-size:12px; color:var(--accent-cyan);">~${formattedDuration}</td>
        <td>
          <button type="button" class="btn btn-secondary btn-sm select-char-row-btn" data-char="${this.escapeHtml(c.name)}" style="padding:4px 12px; font-size:12px; font-weight:600; min-width:85px;">Select</button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    tbody.querySelectorAll(".select-char-row-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const name = btn.getAttribute("data-char");
        this.selectCharacter(name);
      });
    });

    container.style.display = "flex";

    const targetChar = preselectChar || this.selectedCharacter;
    if (targetChar) {
      this.selectCharacter(targetChar);
    }
  }

  selectCharacter(charName, isUserAction = true) {
    const isDifferent = isUserAction && Boolean(this.selectedCharacter && charName && this.selectedCharacter.toUpperCase() !== charName.toUpperCase());
    this.selectedCharacter = charName;

    if (isDifferent) {
      // Different character selected: invalidate downstream steps
      this.alignedClips = [];
      this.selectedClip = null;
      if (this.radar) {
        this.radar.clear(`Selected new character: ${charName}. Ready to search.`);
      }
      this.markStepIncomplete(2);
    }

    const badge = document.getElementById("wizardCharStatsBadge");
    const charObj = this.scriptCharacters.find(c => c.name.toUpperCase() === (charName || "").toUpperCase());

    if (badge) {
      if (charObj) {
        const dur = charObj.estimated_duration_formatted || this.humanizeSpeakingTime(charObj.estimated_duration_sec || (charObj.lines * 3.2));
        badge.innerText = `✓ Selected: ${charObj.name} (${charObj.lines} lines • ~${dur})`;
        badge.style.display = "inline-block";
      } else if (charName) {
        badge.innerText = `✓ Selected: ${charName}`;
        badge.style.display = "inline-block";
      } else {
        badge.style.display = "none";
      }
    }

    // Highlight row and toggle button state
    const tbody = document.getElementById("wizardCharactersTbody");
    if (tbody) {
      tbody.querySelectorAll("tr").forEach(tr => {
        const btn = tr.querySelector(".select-char-row-btn");
        const rowChar = tr.getAttribute("data-char-name") || (btn ? btn.getAttribute("data-char") : "");
        const isMatch = Boolean(charName && rowChar && rowChar.toUpperCase() === charName.toUpperCase());
        if (isMatch) {
          tr.style.backgroundColor = "rgba(16, 185, 129, 0.12)";
          if (btn) {
            btn.className = "btn btn-success btn-sm select-char-row-btn";
            btn.innerHTML = "✓ Selected";
          }
        } else {
          tr.style.backgroundColor = "";
          if (btn) {
            btn.className = "btn btn-secondary btn-sm select-char-row-btn";
            btn.innerHTML = "Select";
          }
        }
      });
    }

    // Sync header dropdown if present
    const headerChar = document.getElementById("headerCharacterSelect");
    if (headerChar) headerChar.value = charName || "";

    this.updateStep1Checklist();
  }

  onEnterStep1() {
    const inputPath = document.getElementById("wizardInputPath");
    const episodeInput = document.getElementById("wizardEpisodeInput");
    const charContainer = document.getElementById("wizardCharacterContainer");
    const episodeBadge = document.getElementById("wizardEpisodeBadge");

    if (inputPath && this.mediaPath && !inputPath.value) {
      inputPath.value = this.mediaPath;
    }
    if (episodeInput && this.episodeCode && !episodeInput.value) {
      episodeInput.value = this.episodeCode;
    }
    if (episodeBadge && this.episodeCode) {
      const m = this.episodeCode.match(/s0?(\d+)[ex]0?(\d+)/i);
      if (m) {
        episodeBadge.innerText = `✓ Detected Season ${parseInt(m[1], 10)} Episode ${parseInt(m[2], 10)}`;
        episodeBadge.style.display = "inline-block";
      }
    }

    if (this.scriptCharacters && this.scriptCharacters.length > 0) {
      if (charContainer) charContainer.style.display = "flex";
      const tbody = document.getElementById("wizardCharactersTbody");
      if (tbody && tbody.children.length === 0) {
        this.renderCharacterTable(this.scriptCharacters, this.selectedCharacter);
      }
    }

    if (this.selectedCharacter) {
      this.selectCharacter(this.selectedCharacter, false);
    }

    this.updateStep1Checklist();
  }

  updateStep1Checklist() {
    const hasMedia = Boolean(this.mediaPath && this.mediaPath.trim().length > 0);
    const hasEpisode = Boolean(this.episodeCode && this.episodeCode.trim().length > 0);
    const hasScript = Boolean(this.scriptLoaded || (this.scriptCharacters && this.scriptCharacters.length > 0));
    const hasChar = Boolean(this.selectedCharacter && this.selectedCharacter.trim().length > 0);

    const setItem = (id, checked) => {
      const el = document.getElementById(id);
      if (el) {
        el.className = `checklist-item ${checked ? 'checked' : ''}`;
        const span = el.querySelector("span");
        if (span) span.innerText = checked ? "✓" : "⚪";
      }
    };

    setItem("chkMedia", hasMedia);
    setItem("chkEpisode", hasEpisode);
    setItem("chkScript", hasScript);
    setItem("chkChar", hasChar);

    const allReady = hasMedia && hasEpisode && hasScript && hasChar;
    const nextBtn = document.getElementById("step1NextBtn");
    if (nextBtn) {
      nextBtn.disabled = !allReady;
    }

    if (allReady) {
      this.markStepCompleted(1);
    } else {
      // Only mark incomplete if Step 1 was not previously completed or required items were cleared
      if (!this.completedSteps.has(1) || !hasMedia || !hasChar) {
        this.markStepIncomplete(1);
      }
    }
  }

  // ----------------- STEP 2: FIND AUDIO (RADAR & WORKERS) -----------------

  setupStep2Events() {
    const backBtn = document.getElementById("step2BackBtn");
    const clearBtn = document.getElementById("clearStep2Btn");
    const startBtn = document.getElementById("startSearchBtn");
    const nextBtn = document.getElementById("step2NextBtn");

    if (backBtn) backBtn.addEventListener("click", () => this.goToStep(1));

    if (clearBtn) {
      clearBtn.addEventListener("click", async () => {
        clearBtn.disabled = true;
        clearBtn.innerText = "⏳ Clearing...";
        try {
          await api.clearStep(2, this.episodeName, this.selectedCharacter);
          this.alignedClips = [];
          this.selectedClip = null;

          if (this.radar) {
            this.radar.clear("Audio cache cleared. Ready to search.");
          }
          this.waveformData = null;

          const durEl = document.getElementById("radarDurationText");
          if (durEl) durEl.innerText = "0:00";

          const startBtn = document.getElementById("startSearchBtn");
          if (startBtn) startBtn.disabled = false;

          const radarClipsEl = document.getElementById("radarClipsCount");
          if (radarClipsEl) radarClipsEl.innerText = "0 clips";

          const cursorTimeEl = document.getElementById("radarCursorTime");
          if (cursorTimeEl) cursorTimeEl.innerText = "0:00";

          // Reset worker telemetry cards
          const sttWorkersCount = parseInt(document.getElementById("step2SttWorkers")?.value) || 2;
          const sttContainer = document.getElementById("radarWorkersContainer");
          if (sttContainer) {
            sttContainer.innerHTML = "";
            for (let i = 1; i <= sttWorkersCount; i++) {
              this.updateSTTWorkerUI({
                worker_id: `worker-stt-${i}`,
                state: "idle",
                snippet: "Awaiting extraction start...",
                chunk_start: 0,
                chunk_end: 0
              });
            }
          }

          const demucsWorkersCount = parseInt(document.getElementById("step2DemucsWorkers")?.value) || 2;
          const demucsContainer = document.getElementById("enhancementWorkersContainer");
          if (demucsContainer) {
            demucsContainer.innerHTML = "";
            for (let i = 1; i <= demucsWorkersCount; i++) {
              this.updateEnhancementWorkerUI({
                worker_id: `worker-demucs-${i}`,
                state: "idle",
                snippet: "Queue empty",
                queue_count: 0
              });
            }
          }

          const sttList = document.getElementById("sttQueueItemsList");
          const demucsList = document.getElementById("demucsQueueItemsList");
          const sttSummary = document.getElementById("sttQueueSummary");
          const demucsSummary = document.getElementById("demucsQueueSummary");
          if (sttList) { sttList.innerHTML = ""; sttList.style.display = "none"; }
          if (demucsList) { demucsList.innerHTML = ""; demucsList.style.display = "none"; }
          if (sttSummary) sttSummary.innerText = "0 items";
          if (demucsSummary) demucsSummary.innerText = "0 items";

          const extStatusEl = document.getElementById("step2ExtractionStatusText");
          if (extStatusEl) extStatusEl.innerText = "Audio cache cleared. Ready to search.";

          const gatingStatusEl = document.getElementById("step2GatingStatus");
          if (gatingStatusEl) gatingStatusEl.innerText = "Audio search cache cleared. Click Start Search to extract.";

          const inspector = document.getElementById("clipInspectorContainer");
          if (inspector) inspector.style.display = "none";
          const rawAudio = document.getElementById("inspectorRawAudio");
          const enhAudio = document.getElementById("inspectorEnhancedAudio");
          if (rawAudio) rawAudio.src = "";
          if (enhAudio) enhAudio.src = "";

          if (nextBtn) nextBtn.disabled = true;
          this.markStepIncomplete(2);

          // Re-sample and load clean media audio waveform with cool spinny sampling animation!
          await this.loadMacroWaveform(true);
        } catch (err) {
          console.error("Failed to clear audio cache:", err);
          alert(`Failed to clear cache: ${err.message}`);
        } finally {
          clearBtn.disabled = false;
          clearBtn.innerText = "🗑️ Clear Audio Cache";
        }
      });
    }

    if (startBtn) {
      startBtn.addEventListener("click", () => this.startAudioSearch());
    }

    const sttWorkersInput = document.getElementById("step2SttWorkers");
    if (sttWorkersInput) {
      sttWorkersInput.addEventListener("input", () => {
        const count = Math.max(1, Math.min(8, parseInt(sttWorkersInput.value) || 2));
        const sttContainer = document.getElementById("radarWorkersContainer");
        if (sttContainer) {
          sttContainer.innerHTML = "";
          for (let i = 1; i <= count; i++) {
            this.updateSTTWorkerUI({
              worker_id: `worker-stt-${i}`,
              state: "idle",
              snippet: "Awaiting extraction start...",
              chunk_start: 0,
              chunk_end: 0
            });
          }
        }
      });
    }

    const demucsWorkersInput = document.getElementById("step2DemucsWorkers");
    if (demucsWorkersInput) {
      demucsWorkersInput.addEventListener("input", () => {
        const count = Math.max(1, Math.min(8, parseInt(demucsWorkersInput.value) || 2));
        const demucsContainer = document.getElementById("enhancementWorkersContainer");
        if (demucsContainer) {
          demucsContainer.innerHTML = "";
          for (let i = 1; i <= count; i++) {
            this.updateEnhancementWorkerUI({
              worker_id: `worker-demucs-${i}`,
              state: "idle",
              snippet: "Queue empty",
              queue_count: 0
            });
          }
        }
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        this.markStepCompleted(2);
        this.goToStep(3);
      });
    }
  }

  async loadMacroWaveform(force = false) {
    if (this.radar) {
      setTimeout(() => {
        this.radar.resize();
        if (this.waveformData) {
          this.radar.setData(this.waveformData);
          if (this.alignedClips && this.alignedClips.length > 0) {
            this.radar.setClips(this.alignedClips);
          }
        }
      }, 50);
    }

    const loadingBanner = document.getElementById("waveformLoadingBanner");
    const epName = this.episodeName || (this.episodeCode ? `Episode_${this.episodeCode}` : "Current_Media");
    const durationEl = document.getElementById("radarDurationText");
    const extStatusEl = document.getElementById("step2ExtractionStatusText");

    if (force || !this.waveformData || this.waveformData.episode !== epName || !this.waveformData.duration) {
      if (loadingBanner) loadingBanner.style.display = "flex";
      if (durationEl) durationEl.innerText = "Sampling...";
      if (extStatusEl && (!this.alignedClips || this.alignedClips.length === 0)) {
        extStatusEl.innerText = "⏳ Sampling media file for audio envelope & duration...";
      }
      if (this.radar) {
        this.radar.setLoading(true, "Sampling media timeline & probing audio channels...");
      }

      try {
        // Guarantee smooth animated feedback so user sees the rotating spinner & radar beam
        const [wf] = await Promise.all([
          api.getWaveform(epName, this.mediaPath),
          new Promise(resolve => setTimeout(resolve, 550))
        ]);

        if (wf) {
          wf.clips = (this.alignedClips && this.alignedClips.length > 0) ? this.alignedClips : (wf.clips || []);
          this.waveformData = wf;
          if (this.radar) {
            this.radar.setData(wf);
          }
          if (wf.duration && durationEl) {
            durationEl.innerText = this.formatTime(wf.duration);
          }
        }
        if (extStatusEl && (!this.alignedClips || this.alignedClips.length === 0)) {
          extStatusEl.innerText = "Audio radar initialized. Ready to find character dialogue.";
        }
      } catch (err) {
        console.warn("Waveform not yet generated for episode:", err);
        if (this.radar) {
          this.radar.clear("Ready to search. Timeline unprobed.");
        }
      } finally {
        if (loadingBanner) loadingBanner.style.display = "none";
      }
    }
  }

  async onEnterStep2() {
    await this.loadMacroWaveform(false);
    await this.checkExistingClips();
  }

  async checkExistingClips() {
    if (!this.selectedCharacter) return;
    try {
      const details = await api.getCharacterDetails(this.selectedCharacter, this.episodeName);
      const clipCount = details?.dataset_stats?.clip_count || (this.alignedClips ? this.alignedClips.length : 0);
      const nextBtn = document.getElementById("step2NextBtn");
      const radarClipsEl = document.getElementById("radarClipsCount");
      const extStatusEl = document.getElementById("step2ExtractionStatusText");
      const gatingStatusEl = document.getElementById("step2GatingStatus");

      if (clipCount > 0) {
        if (radarClipsEl) radarClipsEl.innerText = `${clipCount} clips`;
        const durFormatted = details?.dataset_stats?.total_duration_formatted || this.humanizeSpeakingTime(details?.dataset_stats?.total_duration_sec || (clipCount * 3.2));
        if (extStatusEl) extStatusEl.innerText = `✓ Found ${clipCount} existing neural clips for ${this.selectedCharacter} (${durFormatted})`;
        if (gatingStatusEl) gatingStatusEl.innerText = `✓ Complete! ${clipCount} clips matched and isolated for ${this.selectedCharacter} (${durFormatted}).`;
        if (nextBtn) nextBtn.disabled = false;
        this.markStepCompleted(2);

        // Re-enter state: populate radar, Stage A & Stage B done queues, and inspector
        const clipsToUse = (details?.clips && details.clips.length > 0) ? details.clips : this.alignedClips;
        if (clipsToUse && clipsToUse.length > 0) {
          this.alignedClips = clipsToUse;
          if (this.radar) {
            this.radar.setClips(this.alignedClips);
          }

          // Build Stage B (Demucs) items
          const stageBItems = this.alignedClips.map(c => ({
            timecode: `${this.formatTime(c.start_sec)} ➔ ${this.formatTime(c.end_sec)}`,
            start_sec: c.start_sec,
            end_sec: c.end_sec,
            duration: `${(c.end_sec - c.start_sec).toFixed(1)}s`,
            text: c.text,
            character: c.character || this.selectedCharacter,
            state: "completed",
            status_text: c.enhanced_file ? "Isolated & Denoised" : "Extracted",
            file: c.file,
            enhanced_file: c.enhanced_file,
            confidence: c.confidence
          }));
          this.renderStageBQueueItems(stageBItems);

          // Build Stage A (STT) items
          const stageAItems = this.alignedClips.map(c => ({
            timecode: `${this.formatTime(c.start_sec)} ➔ ${this.formatTime(c.end_sec)}`,
            start_sec: c.start_sec,
            end_sec: c.end_sec,
            duration: `${(c.end_sec - c.start_sec).toFixed(1)}s`,
            text: c.text,
            state: "matched",
            status_text: `${Math.round(c.confidence || 95)}% match`
          }));
          this.renderStageAQueueItems(stageAItems);

          // Update worker telemetry cards to matched / done
          const sttCount = parseInt(document.getElementById("step2SttWorkers")?.value) || 2;
          for (let i = 1; i <= sttCount; i++) {
            this.updateSTTWorkerUI({
              worker_id: `worker-stt-${i}`,
              state: "matched",
              snippet: `✓ ${clipCount} dialogue lines aligned`,
              chunk_start: 0,
              chunk_end: 0
            });
          }

          const demucsCount = parseInt(document.getElementById("step2DemucsWorkers")?.value) || 2;
          for (let i = 1; i <= demucsCount; i++) {
            this.updateEnhancementWorkerUI({
              worker_id: `worker-demucs-${i}`,
              state: "idle",
              snippet: `✓ All ${clipCount} clips neural isolated`,
              queue_count: 0
            });
          }

          // Open inspector for the first clip so user can see raw vs isolated immediately
          if (this.alignedClips.length > 0 && !this.selectedClip) {
            this.handleClipSelected(this.alignedClips[0]);
          }
        }
      } else {
        if (!this.completedSteps.has(2)) {
          if (radarClipsEl) radarClipsEl.innerText = "0 clips";
          if (extStatusEl) extStatusEl.innerText = "No clips extracted yet. Ready to search.";
          if (gatingStatusEl) gatingStatusEl.innerText = "Awaiting audio extraction. Click Start Search to begin.";
          if (nextBtn) nextBtn.disabled = true;
          this.markStepIncomplete(2);
          if (this.radar) {
            this.radar.clips = [];
            this.radar.draw();
          }
        }
      }
    } catch (e) {
      console.warn("Could not check existing clips:", e);
      // Retain existing state if we already have clips or completed step 2
      if (this.alignedClips && this.alignedClips.length > 0) {
        if (this.radar) this.radar.setClips(this.alignedClips);
        const nextBtn = document.getElementById("step2NextBtn");
        if (nextBtn) nextBtn.disabled = false;
        this.markStepCompleted(2);
      }
    }
  }

  async startAudioSearch() {
    const startBtn = document.getElementById("startSearchBtn");
    const statusText = document.getElementById("step2ExtractionStatusText");
    if (!this.mediaPath) {
      alert("Missing media path from Step 1.");
      this.goToStep(1);
      return;
    }

    if (startBtn) startBtn.disabled = true;
    if (statusText) statusText.innerText = "⏳ Queuing divide-and-conquer search...";

    // Activate Stage A and Stage B worker telemetry cards
    const sttWorkersInput = document.getElementById("step2SttWorkers");
    const sttWorkers = sttWorkersInput ? Math.max(1, Math.min(8, parseInt(sttWorkersInput.value) || 2)) : 2;

    const sttContainer = document.getElementById("radarWorkersContainer");
    if (sttContainer) {
      sttContainer.innerHTML = "";
      for (let i = 1; i <= sttWorkers; i++) {
        this.updateSTTWorkerUI({
          worker_id: `worker-stt-${i}`,
          state: "scanning",
          snippet: "Initializing divide-and-conquer speech search...",
          chunk_start: 0,
          chunk_end: 0
        });
      }
    }

    const demucsWorkersInput = document.getElementById("step2DemucsWorkers");
    const demucsWorkers = demucsWorkersInput ? Math.max(1, Math.min(8, parseInt(demucsWorkersInput.value) || 2)) : 2;

    const demucsContainer = document.getElementById("enhancementWorkersContainer");
    if (demucsContainer) {
      demucsContainer.innerHTML = "";
      for (let i = 1; i <= demucsWorkers; i++) {
        this.updateEnhancementWorkerUI({
          worker_id: `worker-demucs-${i}`,
          state: "idle",
          snippet: "Waiting for alignment matches...",
          queue_count: 0
        });
      }
    }

    const sttList = document.getElementById("sttQueueItemsList");
    const demucsList = document.getElementById("demucsQueueItemsList");
    if (sttList) { sttList.innerHTML = ""; sttList.style.display = "none"; }
    if (demucsList) { demucsList.innerHTML = ""; demucsList.style.display = "none"; }

    try {
      const minDurationInput = document.getElementById("step2MinDuration");
      const minDuration = minDurationInput ? parseFloat(minDurationInput.value) || 3.0 : 3.0;

      const enhanceInput = document.getElementById("step2EnhanceToggle");
      const enhance = enhanceInput ? enhanceInput.checked : true;

      const simThreshInput = document.getElementById("step2SimilarityThreshold");
      const similarityThreshold = simThreshInput ? parseFloat(simThreshInput.value) || 55.0 : 55.0;

      const noCacheStt = document.getElementById("step2BypassSttCache")?.checked || false;
      const noCacheAlign = document.getElementById("step2BypassAlignCache")?.checked || false;
      const noCacheAudio = document.getElementById("step2BypassAudioCache")?.checked || false;
      const noCacheEnhance = document.getElementById("step2BypassEnhanceCache")?.checked || false;

      await api.runPipeline({
        input_path: this.mediaPath,
        script_path: this.episodeCode || null,
        characters: [this.selectedCharacter],
        min_duration: minDuration,
        similarity_threshold: similarityThreshold,
        stt_workers: sttWorkers,
        demucs_workers: demucsWorkers,
        enhance: enhance,
        no_cache_stt: noCacheStt,
        no_cache_align: noCacheAlign,
        no_cache_audio: noCacheAudio,
        no_cache_enhance: noCacheEnhance,
        targets: ["all"]
      });
      if (statusText) statusText.innerText = "⚡ Parallel search & neural isolation workers active...";
    } catch (err) {
      alert("Search failed to start: " + err.message);
      if (startBtn) startBtn.disabled = false;
    }
  }

  setupRadar() {
    this.radar = new WaveformRadar("macroWaveformCanvas", {
      onClipSelect: (clip) => this.handleClipSelected(clip),
      onHoverTime: (sec) => {
        const timeEl = document.getElementById("radarCursorTime");
        if (timeEl) timeEl.innerText = this.formatTime(sec);
      }
    });
  }

  handleClipSelected(clip) {
    this.selectedClip = clip;
    const inspector = document.getElementById("clipInspectorContainer");
    if (!inspector) return;
    if (!clip) {
      inspector.style.display = "none";
      return;
    }

    document.getElementById("inspectorClipTitle").innerText = `${clip.character || this.selectedCharacter} • ${this.formatTime(clip.start_sec)} ➔ ${this.formatTime(clip.end_sec)}`;
    document.getElementById("inspectorConfidence").innerText = `${Math.round((clip.confidence || 0.95) * 100)}% Match`;
    document.getElementById("inspectorClipText").innerText = `"${clip.text || 'Dialogue segment'}"`;

    const rawAudio = document.getElementById("inspectorRawAudio");
    const enhAudio = document.getElementById("inspectorEnhancedAudio");

    if (rawAudio && clip.file) {
      rawAudio.src = `/api/v1/audio/stream?path=${encodeURIComponent(clip.file)}`;
    }
    if (enhAudio && clip.enhanced_file) {
      enhAudio.src = `/api/v1/audio/stream?path=${encodeURIComponent(clip.enhanced_file)}`;
    }

    inspector.style.display = "block";
  }

  showClipDetails(clip) {
    this.handleClipSelected(clip);
  }

  // ----------------- STEP 3: MODEL TRAINING -----------------

  setupStep3Events() {
    const backBtn = document.getElementById("step3BackBtn");
    const clearBtn = document.getElementById("clearStep3Btn");
    const nextBtn = document.getElementById("step3NextBtn");

    if (backBtn) backBtn.addEventListener("click", () => this.goToStep(2));

    if (clearBtn) {
      clearBtn.addEventListener("click", async () => {
        if (confirm(`Clear dataset packages and compiled models for ${this.selectedCharacter}?`)) {
          await api.clearStep(3, this.episodeName, this.selectedCharacter);
          await this.onEnterStep3();
          this.markStepIncomplete(3);
        }
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        this.markStepCompleted(3);
        this.goToStep(4);
      });
    }
  }

  async onEnterStep3() {
    document.getElementById("trainingCharName").innerText = this.selectedCharacter || "CLEMENS";
    
    // Load engines status for this character
    try {
      this.engines = await api.getEngines(this.selectedCharacter, this.episodeName);
      this.renderEngineCards(this.engines);
      
      // Update corpus stats
      const details = await api.getCharacterDetails(this.selectedCharacter, this.episodeName);
      if (details && details.dataset_stats) {
        const count = details.dataset_stats.clip_count || 0;
        document.getElementById("trainingClipsCount").innerText = count;
        if (details.dataset_stats.total_duration_formatted) {
          document.getElementById("trainingDurationText").innerText = details.dataset_stats.total_duration_formatted;
        } else {
          const totalSec = details.dataset_stats.total_duration_sec || (count * 3.2);
          document.getElementById("trainingDurationText").innerText = this.humanizeSpeakingTime(totalSec);
        }
      }
    } catch (err) {
      console.warn("Could not load character engine details:", err);
    }
  }

  renderEngineCards(engines) {
    const container = document.getElementById("enginesGridContainer");
    if (!container) return;
    container.innerHTML = "";

    if (!this.activeTrainingEngines) this.activeTrainingEngines = new Set();
    if (!this.trainingProgress) this.trainingProgress = {};
    if (!this.trainingMessage) this.trainingMessage = {};

    let anyTrained = false;

    engines.forEach(eng => {
      if (eng.trained || eng.ready) anyTrained = true;

      const isTraining = this.activeTrainingEngines.has(eng.id);

      const card = document.createElement("div");
      card.className = `engine-card ${isTraining ? 'training-active' : ''}`;
      card.id = `engine_card_${eng.id}`;

      let statusBadgeClass = "badge-locked";
      let statusText = "Needs Configuration";
      let warningHtml = "";

      if (isTraining) {
        statusBadgeClass = "badge-scanning";
        statusText = "⏳ Training in Progress...";
      } else if (!eng.installed) {
        statusBadgeClass = "badge-locked";
        statusText = "Package Missing";
      } else if (eng.id === "piper" && !eng.trainer_installed) {
        statusBadgeClass = "badge-locked";
        statusText = "⚠️ piper-train Missing";
        warningHtml = `
          <div style="background:rgba(245, 158, 11, 0.12); border:1px solid rgba(245, 158, 11, 0.35); border-radius:6px; padding:8px 10px; font-size:11px; color:#fbbf24; margin:8px 0; line-height:1.4;">
            <strong>⚠️ Missing Training Dependency:</strong>
            <code>piper-train</code> CLI is not installed. Piper VITS cannot fine-tune this character's voice without it. Click <strong>Install piper-train</strong> below.
          </div>
        `;
      } else if (eng.id === "piper" && !eng.trained) {
        const f5Eng = engines.find(e => e.id === "f5-tts");
        const f5Ready = f5Eng && (f5Eng.trained || f5Eng.ready);
        warningHtml = `
          <div style="background:rgba(59, 130, 246, 0.12); border:1px solid rgba(59, 130, 246, 0.35); border-radius:6px; padding:8px 10px; font-size:11px; color:#60a5fa; margin:8px 0; line-height:1.4;">
            <strong>ℹ️ Teacher-Student Distillation:</strong>
            Piper fine-tuning uses <strong>F5-TTS</strong> at maximum prosody exaggeration (CFG=5.5) to synthesize an expanded Mark Twain corpus before fine-tuning neural weights.
          </div>
        `;
        statusBadgeClass = "badge-ready";
        statusText = "Ready for Distillation";
      } else if (eng.trained) {
        statusBadgeClass = "badge-ready";
        statusText = "✓ Trained & Ready";
      } else if (eng.ready) {
        statusBadgeClass = "badge-ready";
        statusText = "✓ Zero-Shot Ready";
      } else if (eng.dataset_ready) {
        statusBadgeClass = "badge-ready";
        statusText = "Dataset Ready (Untrained)";
      }

      let actionBtnHtml = "";
      if (isTraining) {
        actionBtnHtml = `
          <button class="btn btn-secondary btn-sm train-engine-btn" data-engine="${eng.id}" disabled style="opacity:0.6; cursor:not-allowed;">
            <span class="spinner" style="width:12px; height:12px; border:2px solid rgba(255,255,255,0.3); border-top-color:#fff; display:inline-block; vertical-align:middle; margin-right:6px;"></span>
            Training ${eng.name}...
          </button>
        `;
      } else if (!eng.installed) {
        actionBtnHtml = `
          <button class="btn btn-secondary btn-sm install-engine-btn" data-engine="${eng.id}">
            📥 Install ${eng.name}
          </button>
        `;
      } else if (eng.id === "piper" && !eng.trainer_installed) {
        actionBtnHtml = `
          <button class="btn btn-warning btn-sm install-engine-btn" data-engine="piper-train" style="background:#d97706; border-color:#b45309; color:#fff; font-weight:600;">
            📥 Install piper-train
          </button>
          <button class="btn btn-secondary btn-sm train-engine-btn" data-engine="${eng.id}">
            🚀 Train ${eng.name}
          </button>
        `;
      } else {
        actionBtnHtml = `
          <button class="btn btn-primary btn-sm train-engine-btn" data-engine="${eng.id}">
            🚀 ${eng.trained ? 'Re-Train / Compile' : 'Train / Prepare'} ${eng.name}
          </button>
        `;
      }

      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
          <div>
            <h3 style="font-size:15px; font-weight:600; color:#fff; margin-bottom:2px;">${eng.name}</h3>
            <div style="font-size:12px; color:var(--text-dim); font-family:var(--font-mono);">${eng.architecture}</div>
          </div>
          <span class="badge ${statusBadgeClass}">${statusText}</span>
        </div>

        <p style="font-size:12px; color:var(--text-muted); line-height:1.4; margin:8px 0;">${eng.description}</p>

        <div style="font-size:11px; color:var(--text-dim); margin-bottom:8px;">
          <strong>Model Path:</strong>
          <div class="model-location-pill" title="${eng.model_path || 'Not yet compiled'}">
            ${eng.model_path || 'Awaiting training / compilation'}
          </div>
        </div>

        ${warningHtml}

        <div class="train-progress-box" id="train_prog_${eng.id}" style="${isTraining ? 'display:block;' : 'display:none;'} margin-bottom:8px;">
          <div class="progress-track"><div class="progress-fill" id="train_fill_${eng.id}" style="width:${(this.trainingProgress && this.trainingProgress[eng.id]) || 25}%;"></div></div>
          <span style="font-size:11px; color:var(--accent-cyan);" id="train_msg_${eng.id}">
            ${(this.trainingMessage && this.trainingMessage[eng.id]) || 'Processing training...'}
          </span>
        </div>

        <div style="display:flex; justify-content:flex-end; gap:8px;">
          ${actionBtnHtml}
        </div>
      `;

      container.appendChild(card);
    });

    // Wire action buttons
    container.querySelectorAll(".install-engine-btn").forEach(btn => {
      btn.addEventListener("click", () => this.handleInstallEngine(btn.getAttribute("data-engine"), btn));
    });

    container.querySelectorAll(".train-engine-btn").forEach(btn => {
      if (!btn.disabled) {
        btn.addEventListener("click", () => this.handleTrainEngine(btn.getAttribute("data-engine"), btn));
      }
    });

    // Check gating
    const nextBtn = document.getElementById("step3NextBtn");
    const statusText = document.getElementById("step3GatingStatus");
    if (anyTrained) {
      if (nextBtn) nextBtn.disabled = false;
      if (statusText) statusText.innerText = "✓ At least one model is trained and ready for synthesis.";
      this.markStepCompleted(3);
    } else if (this.completedSteps.has(3)) {
      if (nextBtn) nextBtn.disabled = false;
      if (statusText) statusText.innerText = "✓ Voice model ready for synthesis.";
    } else {
      if (nextBtn) nextBtn.disabled = true;
      if (statusText) statusText.innerText = "Compile or train at least one voice model to proceed.";
      this.markStepIncomplete(3);
    }
  }

  async handleInstallEngine(engineId, btn) {
    btn.disabled = true;
    const origText = btn.innerText;
    btn.innerText = `⏳ Installing ${engineId}...`;
    try {
      const res = await api.installEngine(engineId);
      if (res && res.job_id) {
        const poll = setInterval(async () => {
          try {
            const j = await api.getJobStatus(res.job_id);
            if (!j) return;
            if (j.status === "completed") {
              clearInterval(poll);
              btn.innerText = `✓ Installed!`;
              if (this.currentStep === 4) {
                setTimeout(() => this.initStep4(), 1000);
              } else {
                setTimeout(() => this.onEnterStep3(), 1000);
              }
            } else if (j.status === "failed") {
              clearInterval(poll);
              alert(`Installation failed for ${engineId}: ${j.error || j.message}`);
              btn.disabled = false;
              btn.innerText = origText;
            } else {
              btn.innerText = `⏳ ${j.message || 'Installing...'}`;
            }
          } catch (e) {
            console.warn(e);
          }
        }, 1200);
      }
    } catch (err) {
      alert("Installation request failed: " + err.message);
      btn.disabled = false;
      btn.innerText = origText;
    }
  }

  async handleTrainEngine(engineId, btn) {
    const eng = (this.engines || []).find(e => e.id === engineId);
    if (engineId === "piper" && eng && !eng.trainer_installed) {
      alert("Cannot train Piper: 'piper-train' CLI is not installed. Please click 'Install piper-train' first to install the training package.");
      return;
    }

    if (!this.activeTrainingEngines) this.activeTrainingEngines = new Set();
    if (!this.trainingProgress) this.trainingProgress = {};
    if (!this.trainingMessage) this.trainingMessage = {};

    this.activeTrainingEngines.add(engineId);
    this.trainingProgress[engineId] = 20;
    this.trainingMessage[engineId] = `Initializing training for ${engineId.toUpperCase()}...`;

    this.renderEngineCards(this.engines);

    try {
      const res = await api.trainModel({
        character_name: this.selectedCharacter,
        episode_name: this.episodeName,
        engine: engineId
      });
      if (res && res.job_id) {
        this.trackTrainingJob(res.job_id, engineId);
      }
    } catch (err) {
      alert("Training trigger failed: " + err.message);
      this.activeTrainingEngines.delete(engineId);
      delete this.trainingProgress[engineId];
      delete this.trainingMessage[engineId];
      this.renderEngineCards(this.engines);
    }
  }

  trackTrainingJob(jobId, engineId) {
    const pollInterval = setInterval(async () => {
      try {
        const job = await api.getJobStatus(jobId);
        if (!job) return;

        if (job.status === "running" || job.status === "queued") {
          if (!this.trainingProgress) this.trainingProgress = {};
          if (!this.trainingMessage) this.trainingMessage = {};
          this.trainingProgress[engineId] = Math.max(20, job.progress || 20);
          this.trainingMessage[engineId] = job.message || "Training in progress...";

          const fill = document.getElementById(`train_fill_${engineId}`);
          const msg = document.getElementById(`train_msg_${engineId}`);
          if (fill) fill.style.width = `${this.trainingProgress[engineId]}%`;
          if (msg) msg.innerText = this.trainingMessage[engineId];
        } else if (job.status === "completed") {
          clearInterval(pollInterval);
          if (this.activeTrainingEngines) this.activeTrainingEngines.delete(engineId);
          if (this.trainingProgress) delete this.trainingProgress[engineId];
          if (this.trainingMessage) delete this.trainingMessage[engineId];
          await this.onEnterStep3();
        } else if (job.status === "failed" || job.status === "cancelled") {
          clearInterval(pollInterval);
          if (this.activeTrainingEngines) this.activeTrainingEngines.delete(engineId);
          if (this.trainingProgress) delete this.trainingProgress[engineId];
          if (this.trainingMessage) delete this.trainingMessage[engineId];
          alert(`Training failed for ${engineId}: ${job.error || job.message}`);
          await this.onEnterStep3();
        }
      } catch (e) {
        console.warn("Polling error for job:", e);
      }
    }, 1000);
  }

  // ----------------- STEP 4: SYNTHESIS STUDIO -----------------

  setupStep4Events() {
    const backBtn = document.getElementById("step4BackBtn");
    const clearBtn = document.getElementById("clearStep4Btn");
    const synthBatchBtn = document.getElementById("studioSynthesizeBatchBtn");
    const quoteSelect = document.getElementById("studioQuoteSelect");
    const speedRange = document.getElementById("studioSpeedRange");
    const speedVal = document.getElementById("studioSpeedValue");

    if (backBtn) backBtn.addEventListener("click", () => this.goToStep(3));

    if (clearBtn) {
      clearBtn.addEventListener("click", async () => {
        if (confirm(`Clear cached synthesized audio for ${this.selectedCharacter}?`)) {
          await api.clearStep(4, this.episodeName, this.selectedCharacter);
          document.getElementById("multiPlayerGrid").innerHTML = "";
        }
      });
    }

    if (speedRange && speedVal) {
      speedRange.addEventListener("input", (e) => {
        speedVal.innerText = `${parseFloat(e.target.value).toFixed(2)}x`;
      });
    }

    const cfgRange = document.getElementById("studioCfgRange");
    const cfgVal = document.getElementById("studioCfgValue");
    if (cfgRange && cfgVal) {
      cfgRange.addEventListener("input", (e) => {
        cfgVal.innerText = parseFloat(e.target.value).toFixed(2);
      });
    }

    const nfeRange = document.getElementById("studioNfeRange");
    const nfeVal = document.getElementById("studioNfeValue");
    if (nfeRange && nfeVal) {
      nfeRange.addEventListener("input", (e) => {
        nfeVal.innerText = parseInt(e.target.value, 10);
      });
    }

    if (quoteSelect) {
      quoteSelect.addEventListener("change", (e) => {
        const idx = parseInt(e.target.value, 10);
        if (!isNaN(idx) && this.characterQuotes[idx]) {
          const q = this.characterQuotes[idx];
          const textEl = document.getElementById("studioDialogueText");
          if (textEl) textEl.value = q.text;

          // Load original actor audio
          const refAudio = document.getElementById("studioRefAudio");
          const refClipId = document.getElementById("studioRefClipId");
          const refTranscript = document.getElementById("studioRefTranscript");

          if (q.stream_url && refAudio) {
            refAudio.src = q.stream_url;
            refAudio.pause();
            refAudio.currentTime = 0;
          }
          if (refClipId) refClipId.innerText = q.clip_id ? `Stem: ${q.clip_id}` : "";
          if (refTranscript) refTranscript.innerText = `"${q.text}"`;
        }
      });
    }

    if (synthBatchBtn) {
      synthBatchBtn.addEventListener("click", () => this.handleBatchSynthesize());
    }
  }

  async onEnterStep4() {
    // Load quotes and reference prompts
    try {
      const details = await api.getCharacterDetails(this.selectedCharacter, this.episodeName);
      this.characterQuotes = details.quotes || [];
      const quoteSelect = document.getElementById("studioQuoteSelect");

      if (quoteSelect) {
        quoteSelect.innerHTML = '<option value="">-- Choose quote to audition original voice --</option>';
        this.characterQuotes.forEach((q, idx) => {
          const opt = document.createElement("option");
          opt.value = idx;
          opt.innerText = `"${q.text.slice(0, 65)}..."`;
          quoteSelect.appendChild(opt);
        });
      }

      // Populate reference audio prompts dropdown
      this.referencePrompts = details.reference_prompts || [];
      const refSelect = document.getElementById("studioRefAudioSelect");
      if (refSelect) {
        refSelect.innerHTML = '<option value="">Default Reference Audio (High SNR ref.wav)</option>';
        this.referencePrompts.forEach(p => {
          const opt = document.createElement("option");
          opt.value = p.path;
          opt.innerText = `${p.name} (${p.duration}s)`;
          refSelect.appendChild(opt);
        });
      }

      // Sync checkboxes with engine readiness & add install/train action buttons
      const engines = await api.getEngines(this.selectedCharacter, this.episodeName);
      engines.forEach(eng => {
        let chk = null;
        let card = null;
        let actionEl = null;
        if (eng.id === "f5-tts") {
          chk = document.getElementById("checkEngineF5");
          card = document.getElementById("cardCheckF5");
          actionEl = document.getElementById("actionF5");
        } else if (eng.id === "xtts-v2") {
          chk = document.getElementById("checkEngineXTTS");
          card = document.getElementById("cardCheckXTTS");
          actionEl = document.getElementById("actionXTTS");
        } else if (eng.id === "kokoro") {
          chk = document.getElementById("checkEngineKokoro");
          card = document.getElementById("cardCheckKokoro");
          actionEl = document.getElementById("actionKokoro");
        } else if (eng.id === "piper") {
          chk = document.getElementById("checkEnginePiper");
          card = document.getElementById("cardCheckPiper");
          actionEl = document.getElementById("actionPiper");
        }

        if (chk && card && actionEl) {
          const isUsable = eng.installed && (eng.trained || eng.ready);
          chk.disabled = !isUsable;

          if (!eng.installed) {
            chk.checked = false;
            card.style.opacity = "0.65";
            actionEl.innerHTML = `
              <button class="btn btn-secondary btn-sm install-engine-btn" data-engine="${eng.id}" style="width:100%; padding:4px 6px; font-size:11px;">
                📥 Install ${eng.name}
              </button>
            `;
          } else if (eng.id === "piper" && !eng.trainer_installed) {
            chk.checked = false;
            card.style.opacity = "0.65";
            actionEl.innerHTML = `
              <button class="btn btn-warning btn-sm install-engine-btn" data-engine="piper-train" style="width:100%; padding:4px 6px; font-size:11px; background:#d97706; border-color:#b45309; color:#fff; font-weight:600;">
                📥 Install piper-train
              </button>
            `;
          } else if (!isUsable) {
            chk.checked = false;
            card.style.opacity = "0.65";
            actionEl.innerHTML = `
              <button class="btn btn-secondary btn-sm train-goto-step3-btn" data-engine="${eng.id}" style="width:100%; padding:4px 6px; font-size:11px;">
                ⚡ Train in Step 3
              </button>
            `;
          } else {
            card.style.opacity = "1.0";
            actionEl.innerHTML = `<div style="font-size:10px; color:var(--accent-green); text-align:center; padding-top:2px;">✓ Ready for audition</div>`;
          }
        }
      });

      // Bind dynamic install and train buttons in Step 4
      document.querySelectorAll(".engine-checkbox-row .install-engine-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          const targetEngine = btn.getAttribute("data-engine");
          this.handleInstallEngine(targetEngine, btn);
        });
      });
      document.querySelectorAll(".engine-checkbox-row .train-goto-step3-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.renderStep(3);
        });
      });

      // Restore dialogue text if present
      const textArea = document.getElementById("studioDialogueText");
      if (textArea) {
        if (this.dialogueText && !textArea.value) {
          textArea.value = this.dialogueText;
        } else if (textArea.value) {
          this.dialogueText = textArea.value;
        }
      }

      // Restore player cards if grid is empty
      const grid = document.getElementById("multiPlayerGrid");
      if (grid && grid.children.length === 0) {
        if (this.lastSynthesizedList && this.lastSynthesizedList.length > 0) {
          this.lastSynthesizedList.forEach(item => {
            grid.appendChild(this.createAudioPlayerCardElement(item));
          });
        } else if (details?.cached_syntheses && details.cached_syntheses.length > 0) {
          details.cached_syntheses.forEach(item => {
            grid.appendChild(this.createAudioPlayerCardElement(item));
          });
        }
      }
    } catch (err) {
      console.warn("Step 4 init error:", err);
    }
  }

  async handleBatchSynthesize() {
    const text = document.getElementById("studioDialogueText").value.trim();
    const speed = parseFloat(document.getElementById("studioSpeedRange").value) || 1.0;
    const seed = parseInt(document.getElementById("studioSeedInput").value, 10) || 42;
    const cfgStrength = parseFloat(document.getElementById("studioCfgRange")?.value) || 5.0;
    const nfeStep = parseInt(document.getElementById("studioNfeRange")?.value, 10) || 48;
    const refAudioPath = document.getElementById("studioRefAudioSelect")?.value || null;

    if (!text) {
      alert("Please enter dialogue text to synthesize.");
      return;
    }

    // Determine checked engines
    const checkedEngines = [];
    if (document.getElementById("checkEngineKokoro")?.checked) checkedEngines.push("kokoro");
    if (document.getElementById("checkEngineF5")?.checked) checkedEngines.push("f5-tts");
    if (document.getElementById("checkEngineXTTS")?.checked) checkedEngines.push("xtts-v2");
    if (document.getElementById("checkEnginePiper")?.checked) checkedEngines.push("piper");

    if (checkedEngines.length === 0) {
      alert("Please select at least one voice model checkbox to generate.");
      return;
    }

    const synthBtn = document.getElementById("studioSynthesizeBatchBtn");
    synthBtn.disabled = true;
    synthBtn.innerHTML = "⏳ Synthesizing Voice Models...";

    // Prepend new live generating cards so previously generated cards stay visible!
    const grid = document.getElementById("multiPlayerGrid");
    const batchTimestamp = Date.now();
    const kokoroPreset = document.getElementById("studioKokoroVoiceSelect")?.value || "character_custom";

    checkedEngines.slice().reverse().forEach(eng => {
      const info = this.formatEngineDisplay(eng);
      const cardId = `player_card_${eng}_${batchTimestamp}`;
      const card = document.createElement("div");
      card.className = "audio-player-card";
      card.id = cardId;
      card.style.borderColor = info.color;
      card.style.background = "var(--bg-surface-elevated, #161c28)";
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
          <div style="flex:1; min-width:0; padding-right:12px;">
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
              <strong style="color:${info.color}; font-size:14px;">${info.icon} ${info.display}</strong>
              <span class="badge" style="background:rgba(255,255,255,0.08); color:var(--text-main); font-size:10px; padding:2px 6px;">${info.architecture}</span>
            </div>
            <div style="font-size:11px; color:var(--text-dim); margin-top:4px;" id="meta_${cardId}">Synthesizing with <strong>${info.name}</strong>...</div>
            <div style="font-size:12px; color:#e2e8f0; margin:6px 0; padding:6px 8px; background:rgba(0,0,0,0.25); border-left:2px solid ${info.color}; border-radius:4px; font-style:italic; line-height:1.4; word-break:break-word;">
              "${text}"
            </div>
            <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:4px;">
              <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">⚡ Speed: ${speed.toFixed(2)}x</span>
              <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">🎛️ CFG: ${cfgStrength.toFixed(2)}</span>
              <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">🔄 NFE: ${nfeStep} steps</span>
              <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">🎲 Seed: ${seed}</span>
              ${eng === "kokoro" ? `<span class="badge" style="font-size:10px; background:rgba(52,211,153,0.12); color:#34d399;">🌸 Style: ${kokoroPreset}</span>` : ''}
            </div>
          </div>
          <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;" id="actions_${cardId}">
            <span class="badge badge-ready" id="badge_${cardId}">Generating</span>
          </div>
        </div>
        <audio id="audio_${cardId}" controls style="width:100%; margin-top:10px;"></audio>
      `;
      grid.prepend(card);
    });

    try {
      const batchRes = await api.synthesizeBatch({
        character_name: this.selectedCharacter,
        episode_name: this.episodeName,
        engines: checkedEngines,
        text: text,
        speed: speed,
        seed: seed,
        ref_audio_path: refAudioPath,
        cfg_strength: cfgStrength,
        nfe_step: nfeStep,
        voice_preset: kokoroPreset
      });

      // Update player cards with results
      Object.entries(batchRes.results).forEach(([eng, res]) => {
        const cardId = `player_card_${eng}_${batchTimestamp}`;
        const cardEl = document.getElementById(cardId);
        const audioEl = document.getElementById(`audio_${cardId}`);
        const metaEl = document.getElementById(`meta_${cardId}`);
        const badgeEl = document.getElementById(`badge_${cardId}`);
        const actionsEl = document.getElementById(`actions_${cardId}`);
        const info = this.formatEngineDisplay(eng);
        const modelName = res.model_name || info.name;
        const badgeText = res.model_badge || info.badge;

        if (res.status === "success" && audioEl) {
          audioEl.src = res.url;
          if (metaEl) {
            metaEl.innerHTML = `<span style="color:var(--accent-green);">✓ Synthesized</span> • Model: <strong style="color:#fff;">${modelName}</strong> • ${res.duration}s • ${res.samplerate}Hz`;
          }
          if (badgeEl) {
            badgeEl.innerText = `${badgeText} Ready`;
            badgeEl.className = "badge badge-ready";
          }
          if (actionsEl && !actionsEl.querySelector(".delete-synth-btn")) {
            const delBtn = document.createElement("button");
            delBtn.className = "btn btn-secondary btn-sm delete-synth-btn";
            delBtn.title = "Delete this synthesis";
            delBtn.style.cssText = "padding:4px 8px; font-size:12px; color:var(--accent-red); border-color:rgba(239,68,68,0.3); background:rgba(239,68,68,0.1); cursor:pointer;";
            delBtn.innerText = "🗑️";
            delBtn.addEventListener("click", () => this.handleDeleteSynthesis(res.synth_id || res.file_path, cardEl));
            actionsEl.appendChild(delBtn);
          }
        } else {
          if (metaEl) metaEl.innerText = `❌ ${modelName} Error: ${res.error}`;
          if (badgeEl) {
            badgeEl.innerText = `${badgeText} Failed`;
            badgeEl.className = "badge badge-locked";
          }
          if (actionsEl && !actionsEl.querySelector(".delete-synth-btn")) {
            const delBtn = document.createElement("button");
            delBtn.className = "btn btn-secondary btn-sm delete-synth-btn";
            delBtn.title = "Dismiss";
            delBtn.style.cssText = "padding:4px 8px; font-size:12px; color:var(--text-dim); cursor:pointer;";
            delBtn.innerText = "✕";
            delBtn.addEventListener("click", () => cardEl.remove());
            actionsEl.appendChild(delBtn);
          }
        }
      });

      if (batchRes && batchRes.results) {
        this.lastSynthesizedList = this.lastSynthesizedList || [];
        Object.entries(batchRes.results).forEach(([eng, res]) => {
          if (res.status === "success") {
            this.lastSynthesizedList.unshift({
              ...res,
              engine: eng,
              text: text,
              speed: speed,
              seed: seed,
              cfg_strength: cfgStrength,
              nfe_step: nfeStep
            });
          }
        });
        this.markStepCompleted(4);
      }
    } catch (err) {
      alert("Batch synthesis error: " + err.message);
    } finally {
      synthBtn.disabled = false;
      synthBtn.innerHTML = "<span>🎙️</span><span>Synthesize Selected Models</span>";
    }
  }

  formatEngineDisplay(engId) {
    const eng = (engId || "").toLowerCase();
    if (eng.includes("f5")) {
      return {
        id: "f5-tts",
        name: "F5-TTS",
        display: "F5-TTS (Flow-Matching DiT)",
        architecture: "Flow-Matching DiT (24kHz)",
        badge: "F5-TTS",
        icon: "✨",
        color: "var(--accent-cyan)"
      };
    } else if (eng.includes("xtts") || eng.includes("coqui")) {
      return {
        id: "xtts-v2",
        name: "Coqui XTTS-v2",
        display: "Coqui XTTS-v2 (Autoregressive)",
        architecture: "Autoregressive + Latents (24kHz)",
        badge: "XTTS-v2",
        icon: "🎙️",
        color: "var(--accent-purple, #a78bfa)"
      };
    } else if (eng.includes("kokoro") || eng.includes("styletts")) {
      return {
        id: "kokoro",
        name: "Kokoro-82M",
        display: "Kokoro (StyleTTS 2 / 82M)",
        architecture: "Single-Pass StyleTTS 2 (24kHz)",
        badge: "Kokoro 82M",
        icon: "🌸",
        color: "var(--accent-emerald, #34d399)"
      };
    } else if (eng.includes("piper")) {
      return {
        id: "piper",
        name: "Piper VITS",
        display: "Piper (Neural VITS / ONNX)",
        architecture: "Neural VITS (22.05kHz)",
        badge: "Piper VITS",
        icon: "⚡",
        color: "var(--accent-amber, #fbbf24)"
      };
    }
    return {
      id: engId || "tts",
      name: (engId || "TTS").toUpperCase(),
      display: (engId || "TTS").toUpperCase(),
      architecture: "Neural Speech Synthesis",
      badge: (engId || "TTS").toUpperCase(),
      icon: "🔊",
      color: "var(--accent-cyan)"
    };
  }

  createAudioPlayerCardElement(item) {
    const info = this.formatEngineDisplay(item.engine);
    const card = document.createElement("div");
    card.className = "audio-player-card";
    const cardUniqueId = item.synth_id || `synth_${Math.random().toString(36).slice(2, 9)}`;
    card.id = `player_card_${cardUniqueId}`;
    card.style.borderColor = "rgba(255,255,255,0.18)";
    card.style.background = "var(--bg-surface-elevated, #161c28)";

    const modelName = item.model_name || info.name;
    const modelDisplay = item.engine_display || info.display;
    const architecture = item.model_architecture || info.architecture;
    const badgeText = item.model_badge || info.badge;
    const textSnippet = item.text || "Spoken dialogue clip";
    const speedVal = item.speed !== undefined ? Number(item.speed).toFixed(2) : "1.00";
    const cfgVal = item.cfg_strength !== undefined ? Number(item.cfg_strength).toFixed(2) : "2.80";
    const nfeVal = item.nfe_step !== undefined ? item.nfe_step : 32;
    const seedVal = item.seed !== undefined ? item.seed : 42;

    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div style="flex:1; min-width:0; padding-right:12px;">
          <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <strong style="color:${info.color}; font-size:14px;">${info.icon} ${modelDisplay}</strong>
            <span class="badge" style="background:rgba(255,255,255,0.08); color:var(--text-main); font-size:10px; padding:2px 6px;">${architecture}</span>
          </div>
          <div style="font-size:11px; color:var(--text-dim); margin-top:4px;">
            Model: <strong style="color:#fff;">${modelName}</strong> • ${item.duration}s • ${item.samplerate}Hz • Cached Session
          </div>
          <div style="font-size:12px; color:#e2e8f0; margin:6px 0; padding:6px 8px; background:rgba(0,0,0,0.25); border-left:2px solid ${info.color}; border-radius:4px; font-style:italic; line-height:1.4; word-break:break-word;">
            "${textSnippet}"
          </div>
          <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:4px;">
            <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">⚡ Speed: ${speedVal}x</span>
            <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">🎛️ CFG: ${cfgVal}</span>
            <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">🔄 NFE: ${nfeVal} steps</span>
            <span class="badge" style="font-size:10px; background:rgba(255,255,255,0.06); color:var(--text-dim);">🎲 Seed: ${seedVal}</span>
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;">
          <span class="badge badge-ready" style="background:rgba(34, 197, 94, 0.15); border:1px solid rgba(34,197,94,0.3); color:#4ade80;">
            ${badgeText}
          </span>
          <button class="btn btn-secondary btn-sm delete-synth-btn" title="Delete this synthesis" style="padding:4px 8px; font-size:12px; color:var(--accent-red); border-color:rgba(239,68,68,0.3); background:rgba(239,68,68,0.1); cursor:pointer;">
            🗑️
          </button>
        </div>
      </div>
      <audio controls src="${item.url}" style="width:100%; margin-top:10px;"></audio>
    `;

    const delBtn = card.querySelector(".delete-synth-btn");
    delBtn.addEventListener("click", () => this.handleDeleteSynthesis(item.synth_id || item.file_path || item.filename, card));

    return card;
  }

  async handleDeleteSynthesis(identifier, cardElement) {
    if (!confirm("Delete this synthesized audio file and its metadata?")) {
      return;
    }
    try {
      await api.deleteSynthesis(identifier);
      if (cardElement) {
        cardElement.style.transition = "all 0.3s ease";
        cardElement.style.opacity = "0";
        cardElement.style.transform = "translateY(-10px)";
        setTimeout(() => cardElement.remove(), 300);
      }
      if (this.lastSynthesizedList) {
        this.lastSynthesizedList = this.lastSynthesizedList.filter(s => s.synth_id !== identifier && s.file_path !== identifier);
      }
    } catch (err) {
      alert("Failed to delete synthesis: " + err.message);
    }
  }

  renderSynthesizedPlayerCards(results) {
    const grid = document.getElementById("multiPlayerGrid");
    if (!grid) return;
    grid.innerHTML = "";
    Object.entries(results).forEach(([eng, res]) => {
      grid.appendChild(this.createAudioPlayerCardElement({ ...res, engine: eng }));
    });
  }

  renderCachedSynthesesCards(cachedList) {
    const grid = document.getElementById("multiPlayerGrid");
    if (!grid) return;
    grid.innerHTML = "";
    cachedList.forEach(item => {
      grid.appendChild(this.createAudioPlayerCardElement(item));
    });
  }

  // ----------------- WEBSOCKET & JOB EVENTS -----------------

  handleSocketEvent(event) {
    if (event.type === "ws_status") {
      const dot = document.getElementById("wsPulseDot");
      const text = document.getElementById("wsStatusText");
      if (dot && text) {
        if (event.connected) {
          dot.className = "pulse-dot";
          text.innerText = "Studio Online";
        } else {
          dot.className = "pulse-dot offline";
          text.innerText = "Reconnecting...";
        }
      }
    } else if (event.type === "job_update") {
      this.handleJobUpdate(event.job);
    } else if (event.type === "worker_event") {
      this.handleWorkerEvent(event.worker);
    } else if (event.type === "clip_discovered") {
      if (this.radar) {
        this.radar.addClip(event.clip);
      }
    }
  }

  handleJobUpdate(job) {
    const banner = document.getElementById("activeJobBanner");
    if (!banner) return;

    if (job.status === "running" || job.status === "queued") {
      banner.classList.add("active");
      document.getElementById("jobTitleText").innerText = `[${job.stage.toUpperCase()}] ${job.message}`;
      document.getElementById("jobProgressFill").style.width = `${job.progress}%`;
      document.getElementById("jobPercentText").innerText = `${Math.round(job.progress)}%`;
      this.activeJobId = job.job_id;

      // Real-time engine training progress card feedback
      if (job.job_type === "train") {
        const eng = (job.params?.engine || "").toLowerCase();
        let targetEngId = null;
        if (eng.includes("piper") || eng.includes("onnx")) targetEngId = "piper";
        else if (eng.includes("xtts")) targetEngId = "xtts-v2";
        else if (eng.includes("f5")) targetEngId = "f5-tts";

        if (targetEngId) {
          if (!this.activeTrainingEngines) this.activeTrainingEngines = new Set();
          this.activeTrainingEngines.add(targetEngId);
          if (!this.trainingProgress) this.trainingProgress = {};
          if (!this.trainingMessage) this.trainingMessage = {};
          this.trainingProgress[targetEngId] = Math.max(20, job.progress || 20);
          this.trainingMessage[targetEngId] = job.message || "Training in progress...";

          const fill = document.getElementById(`train_fill_${targetEngId}`);
          const msg = document.getElementById(`train_msg_${targetEngId}`);
          if (fill) fill.style.width = `${this.trainingProgress[targetEngId]}%`;
          if (msg) msg.innerText = this.trainingMessage[targetEngId];
        }
      }
    } else if (job.status === "completed") {
      banner.classList.add("active");
      document.getElementById("jobTitleText").innerText = `✓ ${job.message}`;
      document.getElementById("jobProgressFill").style.width = "100%";
      document.getElementById("jobPercentText").innerText = "100%";

      // If pipeline job finished, advance Step 2
      if (job.job_type === "pipeline") {
        document.getElementById("step2GatingStatus").innerText = "✓ All clips aligned and enhanced!";
        const nextBtn = document.getElementById("step2NextBtn");
        if (nextBtn) nextBtn.disabled = false;
        this.markStepCompleted(2);
      } else if (job.job_type === "train" || job.job_type === "install_engine") {
        if (job.job_type === "train") {
          const eng = (job.params?.engine || "").toLowerCase();
          let targetEngId = null;
          if (eng.includes("piper") || eng.includes("onnx")) targetEngId = "piper";
          else if (eng.includes("xtts")) targetEngId = "xtts-v2";
          else if (eng.includes("f5")) targetEngId = "f5-tts";
          if (targetEngId && this.activeTrainingEngines) {
            this.activeTrainingEngines.delete(targetEngId);
            if (this.trainingProgress) delete this.trainingProgress[targetEngId];
            if (this.trainingMessage) delete this.trainingMessage[targetEngId];
          }
        }
        // Re-render Step 3 cards
        this.onEnterStep3();
      }

      setTimeout(() => banner.classList.remove("active"), 3500);
    } else if (job.status === "failed") {
      banner.classList.add("active");
      document.getElementById("jobTitleText").innerText = `❌ Error: ${job.error || job.message}`;
      document.getElementById("jobProgressFill").style.background = "var(--accent-rose)";

      if (job.job_type === "train") {
        const eng = (job.params?.engine || "").toLowerCase();
        let targetEngId = null;
        if (eng.includes("piper") || eng.includes("onnx")) targetEngId = "piper";
        else if (eng.includes("xtts")) targetEngId = "xtts-v2";
        else if (eng.includes("f5")) targetEngId = "f5-tts";
        if (targetEngId && this.activeTrainingEngines) {
          this.activeTrainingEngines.delete(targetEngId);
          if (this.trainingProgress) delete this.trainingProgress[targetEngId];
          if (this.trainingMessage) delete this.trainingMessage[targetEngId];
        }
        this.onEnterStep3();
      }
    }
  }

  handleWorkerEvent(worker) {
    if (!worker || !worker.worker_id) return;

    if (worker.worker_id.startsWith("worker-demucs") || worker.worker_id === "worker-demucs-1") {
      // Stage B: Demucs Enhancement worker
      this.updateEnhancementWorkerUI(worker);
      if (worker.queue_items) {
        this.renderStageBQueueItems(worker.queue_items);
      }
    } else {
      // Stage A: STT Search worker
      if (this.radar) {
        this.radar.updateWorker(worker);
      }
      this.updateSTTWorkerUI(worker);
      if (worker.queue_items) {
        this.renderStageAQueueItems(worker.queue_items);
      }
    }
  }

  renderStageAQueueItems(items) {
    const container = document.getElementById("sttQueueItemsList");
    const summary = document.getElementById("sttQueueSummary");
    if (!container || !Array.isArray(items) || items.length === 0) return;

    container.style.display = "flex";

    // Partition items into active/queued and done (matched/unmatched)
    const queueItems = items.filter(it => it.state === "scanning" || it.state === "pending" || it.state === "idle");
    // Place currently scanning item at top of queue
    queueItems.sort((a, b) => (b.state === "scanning" ? 1 : 0) - (a.state === "scanning" ? 1 : 0));

    const doneItems = items.filter(it => it.state === "matched" || it.state === "completed" || it.state === "unmatched");
    const reversedDone = [...doneItems].reverse();

    const matchedCount = items.filter(it => it.state === "matched" || it.state === "completed").length;
    const unmatchedCount = items.filter(it => it.state === "unmatched").length;

    if (summary) {
      summary.innerText = `${matchedCount} matched • ${queueItems.length} queued`;
    }

    container.innerHTML = "";

    // 1. ACTIVE QUEUE (Items at top of queue stay at top!)
    if (queueItems.length > 0) {
      const qSection = document.createElement("div");
      qSection.style.cssText = "display:flex; flex-direction:column; gap:6px;";

      const qHeader = document.createElement("div");
      qHeader.className = "queue-section-header";
      qHeader.innerHTML = `
        <span style="display:flex; align-items:center; gap:6px;">
          <span class="spinner" style="width:10px; height:10px; border:2px solid rgba(6,182,212,0.3); border-top-color:var(--accent-cyan);"></span>
          <span>Queue (${queueItems.length} pending)</span>
        </span>
        <span style="font-size:10px; color:var(--text-dim);">Top of queue</span>
      `;
      qSection.appendChild(qHeader);

      queueItems.forEach(item => {
        qSection.appendChild(this._createStageAItemElement(item));
      });
      container.appendChild(qSection);
    }

    // 2. DONE LIST (Moved here after finishing, with matched or unmatched status)
    if (reversedDone.length > 0) {
      const dSection = document.createElement("div");
      dSection.style.cssText = "display:flex; flex-direction:column; gap:6px; margin-top:8px; border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;";

      const dHeader = document.createElement("div");
      dHeader.className = "queue-section-header";
      dHeader.innerHTML = `
        <span style="display:flex; align-items:center; gap:6px;">
          <span>✓ Done (${reversedDone.length})</span>
        </span>
        <span style="display:flex; gap:8px; font-size:10px;">
          <span style="color:var(--accent-emerald);">${matchedCount} matched</span>
          ${unmatchedCount > 0 ? `<span style="color:var(--text-muted);">${unmatchedCount} unmatched</span>` : ''}
        </span>
      `;
      dSection.appendChild(dHeader);

      reversedDone.forEach(item => {
        dSection.appendChild(this._createStageAItemElement(item));
      });
      container.appendChild(dSection);
    }
  }

  _createStageAItemElement(item) {
    const itemEl = document.createElement("div");
    let stateClass = "";
    let icon = "⏳";
    let badgeClass = "idle";
    let badgeLabel = "queued";

    if (item.state === "scanning") {
      stateClass = "active-scan";
      icon = "⚡";
      badgeClass = "scanning";
      badgeLabel = "scanning";
    } else if (item.state === "matched" || item.state === "completed") {
      stateClass = "completed";
      icon = "✓";
      badgeClass = "matched";
      badgeLabel = "matched";
    } else if (item.state === "unmatched") {
      stateClass = "unmatched";
      icon = "⚪";
      badgeClass = "unmatched";
      badgeLabel = "unmatched";
    }

    itemEl.className = `queue-item ${stateClass}`;
    const textPreview = (item.text || "").trim();
    const shortText = textPreview.length > 34 ? textPreview.slice(0, 34) + "..." : textPreview;

    itemEl.innerHTML = `
      <div class="queue-item-header">
        <div class="queue-item-title" title="${this.escapeHtml(textPreview)}">
          <span>${icon}</span>
          <span style="font-weight:600; color:#fff;">Line ${item.index !== undefined ? item.index + 1 : ''}</span>
          <span>"${this.escapeHtml(shortText)}"</span>
        </div>
        <span class="worker-badge ${badgeClass}">${badgeLabel}</span>
      </div>
      <div class="queue-item-meta">
        <span>${item.start_sec !== undefined && item.end_sec !== undefined ? `${this.formatTime(item.start_sec)} - ${this.formatTime(item.end_sec)} (${item.duration || ''})` : (item.status_text || (item.state === 'unmatched' ? 'Unmatched in timeline' : 'Queued'))}</span>
        <span>${item.confidence ? `${Math.round(item.confidence)}% conf` : ''}</span>
      </div>
    `;

    if (item.start_sec !== undefined && item.end_sec !== undefined && this.radar) {
      itemEl.onclick = () => {
        this.radar.highlightSegment(item.start_sec, item.end_sec);
      };
    }
    return itemEl;
  }

  renderStageBQueueItems(items) {
    const container = document.getElementById("demucsQueueItemsList");
    const summary = document.getElementById("demucsQueueSummary");
    const queueCountEl = document.getElementById("demucsQueueCount");
    if (!container || !Array.isArray(items) || items.length === 0) return;

    container.style.display = "flex";

    // Active Queue: items isolating or waiting
    const queueItems = items.filter(it => it.state === "enhancing" || it.state === "pending" || it.state === "idle");
    queueItems.sort((a, b) => (b.state === "enhancing" ? 1 : 0) - (a.state === "enhancing" ? 1 : 0));

    // Done items: isolated, cached, or unmatched/failed
    const doneItems = items.filter(it => it.state === "matched" || it.state === "completed" || it.state === "unmatched" || it.state === "failed");
    const reversedDone = [...doneItems].reverse();

    const isolatedCount = items.filter(it => it.state === "matched" || it.state === "completed").length;
    const failedCount = items.filter(it => it.state === "unmatched" || it.state === "failed").length;

    if (summary) {
      summary.innerText = `${isolatedCount}/${items.length} isolated • ${queueItems.length} queued`;
    }
    if (queueCountEl) {
      queueCountEl.innerText = `${queueItems.length} pending • ${isolatedCount} done`;
    }

    container.innerHTML = "";

    // 1. ACTIVE QUEUE (Items at top of queue stay at top!)
    if (queueItems.length > 0) {
      const qSection = document.createElement("div");
      qSection.style.cssText = "display:flex; flex-direction:column; gap:6px;";

      const qHeader = document.createElement("div");
      qHeader.className = "queue-section-header";
      qHeader.innerHTML = `
        <span style="display:flex; align-items:center; gap:6px;">
          <span class="spinner" style="width:10px; height:10px; border:2px solid rgba(6,182,212,0.3); border-top-color:var(--accent-cyan);"></span>
          <span>Demucs Queue (${queueItems.length} pending)</span>
        </span>
        <span style="font-size:10px; color:var(--text-dim);">Top of queue</span>
      `;
      qSection.appendChild(qHeader);

      queueItems.forEach(item => {
        qSection.appendChild(this._createStageBItemElement(item));
      });
      container.appendChild(qSection);
    }

    // 2. DONE LIST (Moved here after isolation with status)
    if (reversedDone.length > 0) {
      const dSection = document.createElement("div");
      dSection.style.cssText = "display:flex; flex-direction:column; gap:6px; margin-top:8px; border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;";

      const dHeader = document.createElement("div");
      dHeader.className = "queue-section-header";
      dHeader.innerHTML = `
        <span style="display:flex; align-items:center; gap:6px;">
          <span>✓ Done (${reversedDone.length})</span>
        </span>
        <span style="display:flex; gap:8px; font-size:10px;">
          <span style="color:var(--accent-emerald);">${isolatedCount} isolated</span>
          ${failedCount > 0 ? `<span style="color:var(--accent-rose);">${failedCount} failed</span>` : ''}
        </span>
      `;
      dSection.appendChild(dHeader);

      reversedDone.forEach(item => {
        dSection.appendChild(this._createStageBItemElement(item));
      });
      container.appendChild(dSection);
    }
  }

  _createStageBItemElement(item) {
    const itemEl = document.createElement("div");
    let stateClass = "";
    let icon = "⏳";
    let badgeClass = "idle";
    let badgeLabel = "queued";

    if (item.state === "enhancing") {
      stateClass = "active";
      icon = "⚡";
      badgeClass = "enhancing";
      badgeLabel = "isolating";
    } else if (item.state === "matched" || item.state === "completed") {
      stateClass = "completed";
      icon = "✓";
      badgeClass = "matched";
      badgeLabel = (item.status_text && item.status_text.includes("Cached")) ? "cached" : "isolated";
    } else if (item.state === "unmatched" || item.state === "failed") {
      stateClass = "unmatched";
      icon = "❌";
      badgeClass = "unmatched";
      badgeLabel = "unmatched";
    }

    itemEl.className = `queue-item ${stateClass}`;
    const textPreview = (item.text || "").trim();
    const shortText = textPreview.length > 32 ? textPreview.slice(0, 32) + "..." : textPreview;

    itemEl.innerHTML = `
      <div class="queue-item-header">
        <div class="queue-item-title" title="${this.escapeHtml(textPreview)}">
          <span>${icon}</span>
          <span style="font-weight:600; color:#fff;">Clip ${item.timecode || ''}</span>
          <span>"${this.escapeHtml(shortText)}"</span>
        </div>
        <span class="worker-badge ${badgeClass}">${badgeLabel}</span>
      </div>
      <div class="queue-item-meta">
        <span>${item.timecode || (item.start_sec !== undefined ? `${this.formatTime(item.start_sec)} - ${this.formatTime(item.end_sec)}` : '')} ${item.duration ? `(${item.duration})` : ''}</span>
        <span>${item.status_text || ''}</span>
      </div>
    `;

    itemEl.onclick = () => {
      if (item.start_sec !== undefined && this.radar) {
        this.radar.highlightSegment(item.start_sec, item.end_sec);
      }
      if (item.file || item.enhanced_file) {
        this.showClipDetails({
          character: item.character || this.selectedCharacter,
          text: item.text,
          start_sec: item.start_sec,
          end_sec: item.end_sec,
          confidence: item.confidence || 100,
          file: item.file,
          enhanced_file: item.enhanced_file
        });
      }
    };
    return itemEl;
  }

  updateSTTWorkerUI(worker) {
    const container = document.getElementById("radarWorkersContainer");
    if (!container) return;

    let card = document.getElementById(`sttWorkerCard-${worker.worker_id}`);
    if (!card) {
      card = document.createElement("div");
      card.className = "worker-card";
      card.id = `sttWorkerCard-${worker.worker_id}`;
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600; color:#fff;">${worker.worker_id}</span>
          <span class="worker-badge ${worker.state}" id="sttWorkerBadge-${worker.worker_id}">${worker.state}</span>
        </div>
        <div style="font-size:12px; color:var(--text-muted);" id="sttWorkerSnippet-${worker.worker_id}"></div>
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-dim);" id="sttWorkerTime-${worker.worker_id}"></div>
      `;
      container.appendChild(card);
    }

    const badge = document.getElementById(`sttWorkerBadge-${worker.worker_id}`);
    const snippet = document.getElementById(`sttWorkerSnippet-${worker.worker_id}`);
    const timeEl = document.getElementById(`sttWorkerTime-${worker.worker_id}`);

    if (badge) {
      badge.className = `worker-badge ${worker.state}`;
      badge.innerText = worker.state;
    }
    if (snippet) {
      snippet.innerText = worker.snippet || (worker.state === "scanning" ? "Scanning audio timeline..." : "Matched");
    }
    if (timeEl && worker.chunk_start !== undefined) {
      if (worker.chunk_end > worker.chunk_start) {
        timeEl.innerText = `Timeline: ${this.formatTime(worker.chunk_start)} - ${this.formatTime(worker.chunk_end)}`;
      } else {
        timeEl.innerText = "";
      }
    }
  }

  updateEnhancementWorkerUI(worker) {
    const container = document.getElementById("enhancementWorkersContainer");
    if (!container) return;

    let card = document.getElementById(`demucsWorkerCard-${worker.worker_id}`);
    if (!card) {
      card = document.createElement("div");
      card.className = "worker-card";
      card.id = `demucsWorkerCard-${worker.worker_id}`;
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600; color:#fff;">${worker.worker_id}</span>
          <span class="worker-badge ${worker.state}" id="demucsWorkerBadge-${worker.worker_id}">${worker.state}</span>
        </div>
        <div style="font-size:12px; color:var(--text-muted);" id="demucsWorkerSnippet-${worker.worker_id}"></div>
        <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-dim);" id="demucsWorkerTime-${worker.worker_id}"></div>
      `;
      container.appendChild(card);
    }

    const badge = document.getElementById(`demucsWorkerBadge-${worker.worker_id}`);
    const snippet = document.getElementById(`demucsWorkerSnippet-${worker.worker_id}`);
    const timeEl = document.getElementById(`demucsWorkerTime-${worker.worker_id}`);

    if (badge) {
      badge.className = `worker-badge ${worker.state}`;
      badge.innerText = worker.state;
    }
    if (snippet) {
      snippet.innerText = worker.snippet || (worker.state === "enhancing" ? "Isolating vocal stem..." : "Idle");
    }
    if (timeEl) {
      if (worker.chunk_end > worker.chunk_start) {
        timeEl.innerText = `Timeline: ${this.formatTime(worker.chunk_start)} - ${this.formatTime(worker.chunk_end)}`;
      } else if (worker.queue_count !== undefined && worker.queue_count !== null) {
        timeEl.innerText = `${worker.queue_count} clips pending`;
      } else {
        timeEl.innerText = "";
      }
    }
  }

  // ----------------- AUTO-RESTORATION & DATA -----------------

  async loadSystemStatus() {
    try {
      this.systemStatus = await api.getSystemStatus();
      const pill = document.getElementById("systemAccelPill");
      if (pill && this.systemStatus) {
        const dev = this.systemStatus.device.toUpperCase();
        pill.innerHTML = `⚡ <span>${dev} Acceleration Active</span>`;
      }
    } catch (e) {
      console.warn("Could not load system status:", e);
    }
  }

  async autoRestoreState() {
    try {
      const episodes = await api.getEpisodes();
      if (episodes && episodes.length > 0) {
        // Prefer an episode with existing clips, otherwise top episode
        const activeEp = episodes.find(e => e.clips_count > 0) || episodes[0];
        this.episodeName = activeEp.id;
        
        const m = activeEp.id.match(/s0?(\d+)[ex]0?(\d+)/i);
        const epCode = m ? `s${m[1].padStart(2, '0')}e${m[2].padStart(2, '0')}` : "s06e01";
        this.episodeCode = epCode;

        // Pre-fill Step 1
        const inputPath = document.getElementById("wizardInputPath");
        const epInput = document.getElementById("wizardEpisodeInput");
        if (inputPath && !inputPath.value) {
          inputPath.value = `sftp://elijah@flanopticon.lan/mnt/nas/media/downloads/complete/TV Shows/www.UIndex.org    -    Star Trek The Next Generation S06E01 Times Arrow Part 2 1080p AMZN WEB-DL DDP5 1 H 264-Kitsune/Star Trek The Next Generation S06E01 Times Arrow Part 2 1080p AMZN WEB-DL DDP5 1 H 264-Kitsune.mkv`;
          this.mediaPath = inputPath.value;
        }
        if (epInput && !epInput.value) {
          epInput.value = epCode;
        }

        // Fetch script & characters
        await this.fetchScript(epCode, "startrek", "CLEMENS");

        // Mark Step 1 complete
        this.markStepCompleted(1);

        // Check if clips already exist
        const charDetails = await api.getCharacterDetails("CLEMENS", this.episodeName);
        if (charDetails && charDetails.dataset_stats && charDetails.dataset_stats.clip_count > 0) {
          this.markStepCompleted(2);

          // Check if models exist
          const engines = await api.getEngines("CLEMENS", this.episodeName);
          const anyTrained = engines.some(e => e.trained || e.ready);
          if (anyTrained) {
            this.markStepCompleted(3);
            // Jump user straight to Voice Synthesis Studio or Step 3
            this.goToStep(4);
            return;
          } else {
            this.goToStep(3);
            return;
          }
        }
      }
    } catch (err) {
      console.log("No previous session state to restore:", err);
    }
  }

  formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  humanizeSpeakingTime(seconds) {
    if (!seconds || seconds <= 0) return "0s";
    const sec = Math.round(seconds);
    if (sec < 60) {
      return `${sec}s`;
    }
    const mins = Math.floor(sec / 60);
    const remSec = sec % 60;
    if (mins < 60) {
      return remSec > 0 ? `${mins}m ${remSec}s` : `${mins}m`;
    }
    const hrs = Math.floor(mins / 60);
    const remMin = mins % 60;
    return remMin > 0 ? `${hrs}h ${remMin}m` : `${hrs}h`;
  }

  escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
}

function round(val, dec) {
  return Number(Math.round(val + 'e' + dec) + 'e-' + dec);
}

document.addEventListener("DOMContentLoaded", () => {
  window.app = new VoicesolateWizardApp();
});
