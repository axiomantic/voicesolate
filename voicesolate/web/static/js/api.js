// Voicesolate API & WebSocket Client

class VoicesolateAPI {
  constructor() {
    this.baseUrl = window.location.origin;
    this.ws = null;
    this.subscribers = new Set();
    this.reconnectAttempts = 0;
  }

  async get(endpoint, params = {}) {
    const url = new URL(this.baseUrl + endpoint);
    Object.keys(params).forEach(key => {
      if (params[key] !== undefined && params[key] !== null) {
        url.searchParams.append(key, params[key]);
      }
    });
    const res = await fetch(url.toString());
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Request failed with ${res.status}`);
    }
    return await res.json();
  }

  async post(endpoint, data = {}) {
    const res = await fetch(this.baseUrl + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Request failed with ${res.status}`);
    }
    return await res.json();
  }

  // System
  getSystemStatus() { return this.get("/api/v1/system/status"); }
  getEngines(character, episode) { return this.get("/api/v1/system/engines", { character, episode }); }
  installEngine(engine, params = {}) { return this.post("/api/v1/system/install_engine", { engine, params }); }

  // Episodes & Manifest
  getEpisodes() { return this.get("/api/v1/episodes"); }
  getEpisodeDetails(episodeName) { return this.get(`/api/v1/episodes/${encodeURIComponent(episodeName)}`); }
  getWaveform(episodeName, mediaPath = null) {
    const params = mediaPath ? { media_path: mediaPath } : {};
    return this.get(`/api/v1/episodes/${encodeURIComponent(episodeName)}/waveform`, params);
  }
  getCharacterDetails(characterName, episode) {
    return this.get(`/api/v1/characters/${encodeURIComponent(characterName)}/details`, { episode });
  }

  // Pipeline Actions
  scanMedia(inputPath, scriptPath, provider) {
    return this.post("/api/v1/pipeline/scan", { input_path: inputPath, script_path: scriptPath, provider });
  }

  runPipeline(params) {
    return this.post("/api/v1/pipeline/run", params);
  }

  // Jobs
  getJobs() { return this.get("/api/v1/jobs"); }
  getJobStatus(jobId) { return this.get(`/api/v1/jobs/${encodeURIComponent(jobId)}`); }
  cancelJob(jobId) { return this.post(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`); }

  // Script Detection & Custom Upload
  detectScript(filename) {
    return this.get("/api/v1/scripts/detect", { filename });
  }

  async uploadScript(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(this.baseUrl + "/api/v1/scripts/upload", {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Upload failed with ${res.status}`);
    }
    return await res.json();
  }

  // Model Training
  trainModel(params) {
    return this.post("/api/v1/training/train", params);
  }

  // Wizard Step Clearing
  clearStep(step, episodeName, characterName) {
    return this.post("/api/v1/steps/clear", {
      step: step,
      episode_name: episodeName || null,
      character_name: characterName || null
    });
  }

  // Synthesis
  synthesize(params) {
    return this.post("/api/v1/synthesize", params);
  }

  synthesizeBatch(params) {
    return this.post("/api/v1/synthesize/batch", params);
  }

  deleteSynthesis(identifier) {
    return this.post("/api/v1/synthesis/delete", {
      synth_id: identifier,
      file_path: identifier,
      filename: identifier
    });
  }

  // WebSocket Subscription
  subscribe(callback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  emit(event) {
    this.subscribers.forEach(cb => {
      try { cb(event); } catch (e) { console.error("Subscriber error", e); }
    });
  }

  connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/pipeline`;
    
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log("WebSocket connected to Voicesolate pipeline.");
      this.reconnectAttempts = 0;
      this.emit({ type: "ws_status", connected: true });
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit(data);
      } catch (err) {
        console.error("Malformed WS message:", err);
      }
    };

    this.ws.onclose = () => {
      console.warn("WebSocket closed. Reconnecting...");
      this.emit({ type: "ws_status", connected: false });
      const delay = Math.min(5000, 1000 * Math.pow(1.5, this.reconnectAttempts++));
      setTimeout(() => this.connectWebSocket(), delay);
    };

    this.ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      this.ws.close();
    };
  }
}

export const api = new VoicesolateAPI();
