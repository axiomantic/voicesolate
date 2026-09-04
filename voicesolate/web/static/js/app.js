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
    if (step === 2) {
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

  selectCharacter(charName) {
    this.selectedCharacter = charName;
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

  updateStep1Checklist() {
    const hasMedia = Boolean(this.mediaPath && this.mediaPath.trim().length > 0);
    const hasEpisode = Boolean(this.episodeCode && this.episodeCode.trim().length > 0);
    const hasScript = Boolean(this.scriptLoaded);
    const hasChar = Boolean(this.selectedCharacter && this.selectedCharacter.trim().length > 0);

    const setItem = (id, checked) => {
      const el = document.getElementById(id);
      if (el) {
        el.className = `checklist-item ${checked ? 'checked' : ''}`;
        el.querySelector("span").innerText = checked ? "✓" : "⚪";
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
      this.markStepIncomplete(1);
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
          const sttBadge = document.getElementById("sttWorkerBadge-worker-stt-1");
          const sttSnippet = document.getElementById("sttWorkerSnippet-worker-stt-1");
          const sttTime = document.getElementById("sttWorkerTime-worker-stt-1");
          if (sttBadge) { sttBadge.className = "worker-badge idle"; sttBadge.innerText = "idle"; }
          if (sttSnippet) { sttSnippet.innerText = "Awaiting extraction start..."; }
          if (sttTime) { sttTime.innerText = "--:--"; }

          const demucsBadge = document.getElementById("demucsBadge");
          const demucsSnippet = document.getElementById("demucsSnippet");
          const demucsQueue = document.getElementById("demucsQueueCount");
          if (demucsBadge) { demucsBadge.className = "worker-badge idle"; demucsBadge.innerText = "idle"; }
          if (demucsSnippet) { demucsSnippet.innerText = "Queue empty"; }
          if (demucsQueue) { demucsQueue.innerText = "0 clips pending"; }

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

    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        this.markStepCompleted(2);
        this.goToStep(3);
      });
    }
  }

  async onEnterStep2() {
    if (this.radar) {
      setTimeout(() => this.radar.resize(), 50);
    }

    // Load macro-waveform for this episode if not loaded or duration is 0
    const loadingBanner = document.getElementById("waveformLoadingBanner");
    const epName = this.episodeName || (this.episodeCode ? `Episode_${this.episodeCode}` : "Current_Media");
    const durationEl = document.getElementById("radarDurationText");
    const extStatusEl = document.getElementById("step2ExtractionStatusText");

    if (!this.waveformData || this.waveformData.episode !== epName || !this.waveformData.duration) {
      if (loadingBanner) loadingBanner.style.display = "flex";
      if (durationEl) durationEl.innerText = "Sampling...";
      if (extStatusEl && (!this.alignedClips || this.alignedClips.length === 0)) {
        extStatusEl.innerText = "⏳ Sampling media file for audio envelope & duration...";
      }
      if (this.radar) {
        this.radar.setLoading(true, "Sampling media timeline & probing audio channels...");
      }

      try {
        const wf = await api.getWaveform(epName, this.mediaPath);
        this.waveformData = wf;
        if (this.radar) {
          this.radar.setData(wf);
        }
        if (wf && wf.duration) {
          if (durationEl) durationEl.innerText = this.formatTime(wf.duration);
        }
        if (extStatusEl && (!this.alignedClips || this.alignedClips.length === 0)) {
          extStatusEl.innerText = "Audio radar initialized. Ready to find character dialogue.";
        }
      } catch (err) {
        console.warn("Waveform not yet generated for episode:", err);
        if (this.radar) {
          this.radar.setLoading(false);
        }
      } finally {
        if (loadingBanner) loadingBanner.style.display = "none";
      }
    }

    // Check if character already has clips in output
    await this.checkExistingClips();
  }

  async checkExistingClips() {
    if (!this.selectedCharacter) return;
    try {
      const details = await api.getCharacterDetails(this.selectedCharacter, this.episodeName);
      const clipCount = details?.dataset_stats?.clip_count || 0;
      const nextBtn = document.getElementById("step2NextBtn");
      const radarClipsEl = document.getElementById("radarClipsCount");
      const extStatusEl = document.getElementById("step2ExtractionStatusText");
      const gatingStatusEl = document.getElementById("step2GatingStatus");

      if (clipCount > 0) {
        if (radarClipsEl) radarClipsEl.innerText = `${clipCount} clips`;
        if (extStatusEl) extStatusEl.innerText = `✓ Found ${clipCount} existing neural clips for ${this.selectedCharacter}`;
        if (gatingStatusEl) gatingStatusEl.innerText = `✓ Complete! ${clipCount} clips matched and isolated for ${this.selectedCharacter}.`;
        if (nextBtn) nextBtn.disabled = false;
        this.markStepCompleted(2);
      } else {
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
    } catch (e) {
      const nextBtn = document.getElementById("step2NextBtn");
      const radarClipsEl = document.getElementById("radarClipsCount");
      const extStatusEl = document.getElementById("step2ExtractionStatusText");
      const gatingStatusEl = document.getElementById("step2GatingStatus");
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
    const sttBadge = document.getElementById("sttWorkerBadge-worker-stt-1");
    const sttSnippet = document.getElementById("sttWorkerSnippet-worker-stt-1");
    const sttTime = document.getElementById("sttWorkerTime-worker-stt-1");
    if (sttBadge) { sttBadge.className = "worker-badge scanning"; sttBadge.innerText = "scanning"; }
    if (sttSnippet) { sttSnippet.innerText = "Initializing divide-and-conquer speech search..."; }
    if (sttTime) { sttTime.innerText = "Timeline: 0:00..."; }

    const demucsBadge = document.getElementById("demucsBadge");
    const demucsSnippet = document.getElementById("demucsSnippet");
    const demucsQueue = document.getElementById("demucsQueueCount");
    if (demucsBadge) { demucsBadge.className = "worker-badge idle"; demucsBadge.innerText = "queued"; }
    if (demucsSnippet) { demucsSnippet.innerText = "Waiting for alignment matches..."; }
    if (demucsQueue) { demucsQueue.innerText = "Queue waiting"; }

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
        const totalSec = details.dataset_stats.total_duration_sec || (count * 3.2);
        document.getElementById("trainingDurationText").innerText = this.humanizeSpeakingTime(totalSec);
      }
    } catch (err) {
      console.warn("Could not load character engine details:", err);
    }
  }

  renderEngineCards(engines) {
    const container = document.getElementById("enginesGridContainer");
    if (!container) return;
    container.innerHTML = "";

    let anyTrained = false;

    engines.forEach(eng => {
      if (eng.trained || eng.ready) anyTrained = true;

      const card = document.createElement("div");
      card.className = "engine-card";
      card.id = `engine_card_${eng.id}`;

      let statusBadgeClass = "badge-locked";
      let statusText = "Needs Configuration";
      if (!eng.installed) {
        statusBadgeClass = "badge-locked";
        statusText = "Package Missing";
      } else if (eng.trained || eng.ready) {
        statusBadgeClass = "badge-ready";
        statusText = "✓ Trained & Ready";
      } else if (eng.dataset_ready) {
        statusBadgeClass = "badge-ready";
        statusText = "Dataset Ready";
      }

      let actionBtnHtml = "";
      if (!eng.installed) {
        actionBtnHtml = `
          <button class="btn btn-secondary btn-sm install-engine-btn" data-engine="${eng.id}">
            📥 Install ${eng.name}
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

        <div class="train-progress-box" id="train_prog_${eng.id}" style="display:none; margin-bottom:8px;">
          <div class="progress-track"><div class="progress-fill" style="width:50%;"></div></div>
          <span style="font-size:11px; color:var(--accent-cyan);">Processing training...</span>
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
      btn.addEventListener("click", () => this.handleTrainEngine(btn.getAttribute("data-engine"), btn));
    });

    // Check gating
    const nextBtn = document.getElementById("step3NextBtn");
    const statusText = document.getElementById("step3GatingStatus");
    if (anyTrained) {
      if (nextBtn) nextBtn.disabled = false;
      if (statusText) statusText.innerText = "✓ At least one model is trained and ready for synthesis.";
      this.markStepCompleted(3);
    } else {
      if (nextBtn) nextBtn.disabled = true;
      if (statusText) statusText.innerText = "Compile or train at least one voice model to proceed.";
      this.markStepIncomplete(3);
    }
  }

  async handleInstallEngine(engineId, btn) {
    btn.disabled = true;
    btn.innerText = `⏳ Installing ${engineId}...`;
    try {
      await api.installEngine(engineId);
    } catch (err) {
      alert("Installation request failed: " + err.message);
      btn.disabled = false;
    }
  }

  async handleTrainEngine(engineId, btn) {
    btn.disabled = true;
    btn.innerText = `⏳ Training ${engineId}...`;
    try {
      await api.trainModel({
        character_name: this.selectedCharacter,
        episode_name: this.episodeName,
        engine: engineId
      });
    } catch (err) {
      alert("Training trigger failed: " + err.message);
      btn.disabled = false;
    }
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
            refAudio.play().catch(() => {});
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

      // Sync checkboxes with engine readiness
      const engines = await api.getEngines(this.selectedCharacter, this.episodeName);
      engines.forEach(eng => {
        let chk = null;
        let card = null;
        if (eng.id === "f5-tts") {
          chk = document.getElementById("checkEngineF5");
          card = document.getElementById("cardCheckF5");
        } else if (eng.id === "xtts-v2") {
          chk = document.getElementById("checkEngineXTTS");
          card = document.getElementById("cardCheckXTTS");
        } else if (eng.id === "piper") {
          chk = document.getElementById("checkEnginePiper");
          card = document.getElementById("cardCheckPiper");
        }

        if (chk && card) {
          const isUsable = eng.installed && (eng.trained || eng.ready);
          chk.disabled = !isUsable;
          if (!isUsable) {
            chk.checked = false;
            card.style.opacity = "0.5";
            card.title = `${eng.name} is not trained or installed. Train it in Step 3.`;
          } else {
            card.style.opacity = "1.0";
          }
        }
      });
    } catch (err) {
      console.warn("Step 4 init error:", err);
    }
  }

  async handleBatchSynthesize() {
    const text = document.getElementById("studioDialogueText").value.trim();
    const speed = parseFloat(document.getElementById("studioSpeedRange").value) || 1.0;
    const seed = parseInt(document.getElementById("studioSeedInput").value, 10) || 42;

    if (!text) {
      alert("Please enter dialogue text to synthesize.");
      return;
    }

    // Determine checked engines
    const checkedEngines = [];
    if (document.getElementById("checkEngineF5").checked) checkedEngines.push("f5-tts");
    if (document.getElementById("checkEngineXTTS").checked) checkedEngines.push("xtts-v2");
    if (document.getElementById("checkEnginePiper").checked) checkedEngines.push("piper");

    if (checkedEngines.length === 0) {
      alert("Please select at least one voice model checkbox to generate.");
      return;
    }

    const synthBtn = document.getElementById("studioSynthesizeBatchBtn");
    synthBtn.disabled = true;
    synthBtn.innerHTML = "⏳ Synthesizing Voice Models...";

    // Setup player cards in grid
    const grid = document.getElementById("multiPlayerGrid");
    grid.innerHTML = "";

    checkedEngines.forEach(eng => {
      const card = document.createElement("div");
      card.className = "audio-player-card";
      card.id = `player_card_${eng}`;
      card.style.borderColor = "var(--accent-cyan)";
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong style="color:var(--accent-cyan); font-size:13px;">✨ ${eng.toUpperCase()} AI Voice</strong>
            <span style="font-size:11px; color:var(--text-dim);" id="meta_${eng}">Synthesizing...</span>
          </div>
          <span class="badge badge-ready" id="badge_${eng}">Generating</span>
        </div>
        <audio id="audio_${eng}" controls style="width:100%; margin-top:8px;"></audio>
      `;
      grid.appendChild(card);
    });

    try {
      const batchRes = await api.synthesizeBatch({
        character_name: this.selectedCharacter,
        episode_name: this.episodeName,
        engines: checkedEngines,
        text: text,
        speed: speed,
        seed: seed
      });

      // Update player cards with results
      Object.entries(batchRes.results).forEach(([eng, res]) => {
        const audioEl = document.getElementById(`audio_${eng}`);
        const metaEl = document.getElementById(`meta_${eng}`);
        const badgeEl = document.getElementById(`badge_${eng}`);

        if (res.status === "success" && audioEl) {
          audioEl.src = res.url;
          if (metaEl) metaEl.innerText = `✓ Generated ${res.duration}s (${res.samplerate}Hz)`;
          if (badgeEl) {
            badgeEl.innerText = "Ready";
            badgeEl.className = "badge badge-ready";
          }
        } else {
          if (metaEl) metaEl.innerText = `❌ Error: ${res.error}`;
          if (badgeEl) {
            badgeEl.innerText = "Failed";
            badgeEl.className = "badge badge-locked";
          }
        }
      });
    } catch (err) {
      alert("Batch synthesis error: " + err.message);
    } finally {
      synthBtn.disabled = false;
      synthBtn.innerHTML = "<span>🎙️</span><span>Synthesize Selected Models</span>";
    }
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
        // Re-render Step 3 cards
        this.onEnterStep3();
      }

      setTimeout(() => banner.classList.remove("active"), 3500);
    } else if (job.status === "failed") {
      banner.classList.add("active");
      document.getElementById("jobTitleText").innerText = `❌ Error: ${job.error || job.message}`;
      document.getElementById("jobProgressFill").style.background = "var(--accent-rose)";
    }
  }

  handleWorkerEvent(worker) {
    if (!worker || !worker.worker_id) return;

    if (worker.worker_id.startsWith("worker-demucs") || worker.worker_id === "worker-demucs-1") {
      // Stage B: Demucs Enhancement worker
      const badge = document.getElementById("demucsBadge");
      const snippet = document.getElementById("demucsSnippet");
      const queueCount = document.getElementById("demucsQueueCount");
      if (badge) {
        badge.className = `worker-badge ${worker.state}`;
        badge.innerText = worker.state;
      }
      if (snippet) {
        snippet.innerText = worker.snippet || "Isolating vocal stem...";
      }
      if (queueCount && worker.queue_count !== undefined && worker.queue_count !== null) {
        queueCount.innerText = `${worker.queue_count} clips pending`;
      }
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
    const matchedCount = items.filter(it => it.state === "matched" || it.state === "completed").length;
    if (summary) {
      summary.innerText = `${matchedCount}/${items.length} matched`;
    }

    container.innerHTML = "";
    items.forEach((item, idx) => {
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
      }

      itemEl.className = `queue-item ${stateClass}`;
      const textPreview = (item.text || "").trim();
      const shortText = textPreview.length > 34 ? textPreview.slice(0, 34) + "..." : textPreview;

      itemEl.innerHTML = `
        <div class="queue-item-header">
          <div class="queue-item-title" title="${this.escapeHtml(textPreview)}">
            <span>${icon}</span>
            <span style="font-weight:600; color:#fff;">#${idx + 1}</span>
            <span>"${this.escapeHtml(shortText)}"</span>
          </div>
          <span class="worker-badge ${badgeClass}">${badgeLabel}</span>
        </div>
        <div class="queue-item-meta">
          <span>${item.start_sec !== undefined && item.end_sec !== undefined ? `${this.formatTime(item.start_sec)} - ${this.formatTime(item.end_sec)} (${item.duration || ''})` : (item.status_text || 'Queued')}</span>
          <span>${item.confidence ? `${Math.round(item.confidence)}% conf` : ''}</span>
        </div>
      `;

      if (item.start_sec !== undefined && item.end_sec !== undefined && this.radar) {
        itemEl.onclick = () => {
          this.radar.highlightSegment(item.start_sec, item.end_sec);
        };
      }
      container.appendChild(itemEl);
    });
  }

  renderStageBQueueItems(items) {
    const container = document.getElementById("demucsQueueItemsList");
    const summary = document.getElementById("demucsQueueSummary");
    const queueCountEl = document.getElementById("demucsQueueCount");
    if (!container || !Array.isArray(items) || items.length === 0) return;

    container.style.display = "flex";
    const completedCount = items.filter(it => it.state === "matched" || it.state === "completed").length;
    const pendingCount = items.filter(it => it.state === "pending" || it.state === "idle").length;

    if (summary) {
      summary.innerText = `${completedCount}/${items.length} isolated`;
    }
    if (queueCountEl) {
      queueCountEl.innerText = `${pendingCount} clips pending (${completedCount}/${items.length} done)`;
    }

    container.innerHTML = "";
    items.forEach((item, idx) => {
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
      }

      itemEl.className = `queue-item ${stateClass}`;
      const textPreview = (item.text || "").trim();
      const shortText = textPreview.length > 32 ? textPreview.slice(0, 32) + "..." : textPreview;

      itemEl.innerHTML = `
        <div class="queue-item-header">
          <div class="queue-item-title" title="${this.escapeHtml(textPreview)}">
            <span>${icon}</span>
            <span style="font-weight:600; color:#fff;">Clip ${idx + 1}</span>
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

      container.appendChild(itemEl);
    });
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
