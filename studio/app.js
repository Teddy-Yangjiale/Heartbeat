const $ = (id) => document.getElementById(id);

const COLORS = {
  song: "#56cce7",
  heartbeat: "#ee6a91",
  vocals: "#edc55a",
  accompaniment: "#7bd88f",
};

const state = {
  jobId: null,
  analysis: null,
  heartbeatSummary: null,
  tracks: new Map(),
  audioContext: null,
  masterGain: null,
  sources: [],
  playing: false,
  startedAt: 0,
  pausedAt: 0,
  duration: 0,
  zoom: 72,
  animationFrame: null,
  phaseTouched: false,
  renderedPhase: 0,
  dirty: false,
};

function setStatus(message, error = false) {
  $("status").textContent = message;
  $("status").style.color = error ? "#ee6a91" : "";
}

function formatTime(seconds) {
  const safe = Math.max(0, seconds || 0);
  const mins = Math.floor(safe / 60).toString().padStart(2, "0");
  const secs = Math.floor(safe % 60).toString().padStart(2, "0");
  const millis = Math.floor((safe % 1) * 1000).toString().padStart(3, "0");
  return `${mins}:${secs}.${millis}`;
}

function dbToGain(db) {
  return Math.pow(10, Number(db) / 20);
}

function currentPosition() {
  if (!state.playing || !state.audioContext) return state.pausedAt;
  return Math.min(state.duration, state.audioContext.currentTime - state.startedAt);
}

function snapTime(seconds) {
  const snap = Number($("snap").value);
  const bpm = Number($("bpm").value) || 120;
  if (!snap) return Math.max(0, seconds);
  const step = (60 / bpm) * snap;
  return Math.max(0, Math.round(seconds / step) * step);
}

function markDirty(message = "Arrangement changed — render to update the exported mix.") {
  if (!state.jobId) return;
  state.dirty = true;
  $("renderNotice").hidden = false;
  $("renderNotice").textContent = message;
  $("render").disabled = false;
  $("saveState").textContent = "EDITED";
}

function markRendered() {
  state.dirty = false;
  state.renderedPhase = Number($("phase").value);
  $("renderNotice").hidden = true;
  $("render").disabled = false;
  $("saveState").textContent = "RENDERED";
}

function bindFileSlot(inputId, labelId, slotId) {
  const input = $(inputId);
  const label = $(labelId);
  const slot = $(slotId);
  input.addEventListener("change", () => {
    label.textContent = input.files[0]?.name || "Drop WAV / MP3";
  });
  ["dragenter", "dragover"].forEach((type) => slot.addEventListener(type, (event) => {
    event.preventDefault();
    slot.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((type) => slot.addEventListener(type, () => slot.classList.remove("dragging")));
}

bindFileSlot("heartbeatFile", "heartbeatName", "heartbeatSlot");
bindFileSlot("songFile", "songName", "songSlot");
initializeBackendStatus();

async function initializeBackendStatus() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    if (!health.demucs_available) {
      $("stems").disabled = true;
      $("stems").parentElement.title = "Install the optional Demucs backend to enable stem separation.";
      setStatus("Ready. Demucs is not installed, so stem separation is disabled; core remixing is available.");
    }
  } catch (_) {
    setStatus("Backend health check failed. Restart run_studio.bat.", true);
  }
}

$("stems").addEventListener("change", () => {
  $("melody").disabled = !$("stems").checked;
  if (!$("stems").checked) $("melody").checked = false;
});

$("heartbeatGain").addEventListener("input", () => {
  $("gainValue").textContent = `${Number($("heartbeatGain").value).toFixed(1)} dB`;
  markDirty();
});

$("phase").addEventListener("input", () => {
  state.phaseTouched = true;
  $("phaseValue").textContent = `${Number($("phase").value).toFixed(3)} s`;
  markDirty("Phase changed — the clip preview is shifted visually; render for phase-vocoder audio.");
  drawTrack(state.tracks.get("heartbeat"));
});

$("bpm").addEventListener("input", () => {
  $("autoBpm").checked = false;
  markDirty("BPM changed — render to stretch the heartbeat loop to the new grid.");
  drawAll();
});
$("meter").addEventListener("change", drawAll);
$("snap").addEventListener("change", drawAll);

$("analyze").addEventListener("click", analyzeSession);
$("render").addEventListener("click", renderSession);
$("play").addEventListener("click", togglePlayback);
$("stop").addEventListener("click", stopPlayback);
$("toStart").addEventListener("click", () => seek(0));
$("zoom").addEventListener("input", () => { state.zoom = Number($("zoom").value); drawAll(); });
$("zoomIn").addEventListener("click", () => adjustZoom(12));
$("zoomOut").addEventListener("click", () => adjustZoom(-12));

window.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); togglePlayback(); }
  if (event.code === "Home") { event.preventDefault(); seek(0); }
});
window.addEventListener("resize", drawAll);

async function analyzeSession() {
  const heartbeat = $("heartbeatFile").files[0];
  const song = $("songFile").files[0];
  if (!heartbeat || !song) {
    setStatus("Choose both heartbeat and song audio first.", true);
    return;
  }
  stopPlayback();
  $("analyze").disabled = true;
  $("progress").hidden = false;
  setStatus("Analyzing heartbeat, song beat grid, and loop alignment…");
  const form = new FormData();
  form.append("heartbeat", heartbeat);
  form.append("song", song);
  form.append("beats_per_bar", $("meter").value);
  form.append("loop_beats", $("loopBeats").value);
  form.append("heartbeat_gain_db", $("heartbeatGain").value);
  form.append("separate_stems", $("stems").checked);
  form.append("extract_melody", $("melody").checked);
  if (!$("autoBpm").checked) form.append("bpm", $("bpm").value);
  if (state.phaseTouched) form.append("first_beat", $("phase").value);
  try {
    const response = await fetch("/api/process", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed");
    await loadSession(payload, song.name);
    setStatus("Session ready. Inspect confidence, listen, then adjust phase or gain if needed.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    $("analyze").disabled = false;
    $("progress").hidden = true;
  }
}

async function loadSession(payload, projectName) {
  state.jobId = payload.job_id;
  state.analysis = payload.analysis;
  state.heartbeatSummary = payload.heartbeat_summary;
  state.phaseTouched = false;
  $("projectName").textContent = projectName || payload.analysis.filename;
  $("bpm").value = Number(payload.analysis.estimated_bpm).toFixed(2);
  $("autoBpm").checked = !payload.analysis.bpm_overridden;
  $("meter").value = String(payload.analysis.beats_per_bar);
  $("phase").max = Math.max(4, payload.analysis.duration_seconds).toFixed(3);
  $("phase").value = Number(payload.analysis.first_beat_seconds).toFixed(3);
  $("phaseValue").textContent = `${Number(payload.analysis.first_beat_seconds).toFixed(3)} s`;
  state.renderedPhase = Number(payload.analysis.first_beat_seconds);
  state.duration = Number(payload.analysis.duration_seconds);
  updateAnalysisPanel(payload);
  updateDownloads(payload.final_mix_wav_url, payload.final_mix_mp3_url);
  await ensureAudioContext();
  state.tracks.clear();
  const hasStems = payload.tracks.some((track) => track.id === "vocals");
  for (const trackInfo of payload.tracks) {
    const track = {
      ...trackInfo,
      color: COLORS[trackInfo.kind] || "#a0a8b2",
      gainDb: 0,
      pan: 0,
      mute: hasStems && trackInfo.id === "song",
      solo: false,
      buffer: null,
      gainNode: null,
      panNode: null,
      analyser: null,
    };
    state.tracks.set(track.id, track);
  }
  renderTrackShells();
  await Promise.all([...state.tracks.values()].map(loadTrackBuffer));
  $("emptyState").style.display = "none";
  $("playhead").style.display = "block";
  markRendered();
  drawAll();
}

async function ensureAudioContext() {
  if (!state.audioContext) {
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    state.masterGain = state.audioContext.createGain();
    state.masterGain.connect(state.audioContext.destination);
  }
  if (state.audioContext.state === "suspended") await state.audioContext.resume();
}

async function loadTrackBuffer(track) {
  const response = await fetch(track.url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${track.name}`);
  const bytes = await response.arrayBuffer();
  track.buffer = await state.audioContext.decodeAudioData(bytes.slice(0));
  state.duration = Math.max(state.duration, track.buffer.duration);
  drawTrack(track);
}

function renderTrackShells() {
  const list = $("trackList");
  const mixer = $("mixerChannels");
  list.innerHTML = "";
  mixer.innerHTML = "";
  for (const track of state.tracks.values()) {
    const row = document.createElement("div");
    row.className = "track-row";
    row.dataset.track = track.id;
    row.innerHTML = `
      <div class="track-controls" style="--track-color:${track.color}">
        <span class="track-color" style="background:${track.color}"></span>
        <div class="track-title"><span>${track.name}</span><span class="track-buttons"><button class="mute ${track.mute ? "active" : ""}">M</button><button class="solo">S</button></span></div>
        <label class="track-slider"><span>VOL</span><input class="volume" type="range" min="-36" max="6" step="0.1" value="0"><output>0.0</output></label>
        <label class="track-slider"><span>PAN</span><input class="pan" type="range" min="-1" max="1" step="0.01" value="0"><output>C</output></label>
      </div>
      <div class="track-wave ${track.id === "heartbeat" ? "heartbeat-track" : ""}">
        <span class="clip-label">${track.name}</span>${track.id === "heartbeat" ? '<span class="drag-hint">drag to adjust phase</span>' : ""}<canvas></canvas>
      </div>`;
    list.appendChild(row);
    bindTrackControls(row, track);

    const channel = document.createElement("div");
    channel.className = "mixer-channel";
    channel.style.setProperty("--track-color", track.color);
    channel.innerHTML = `<strong>${track.name}</strong><div class="meter"><span></span></div><div class="fader-wrap"><input class="vertical-fader" type="range" min="-36" max="6" step="0.1" value="0"></div><small>0.0 dB</small>`;
    mixer.appendChild(channel);
    const mixerFader = channel.querySelector("input");
    mixerFader.addEventListener("input", () => setTrackGain(track, Number(mixerFader.value), row, channel));
    track.row = row;
    track.channel = channel;
    if (track.id === "heartbeat") bindPhaseDrag(row.querySelector(".track-wave"));
  }
  applyMuteSolo();
}

function bindTrackControls(row, track) {
  row.querySelector(".mute").addEventListener("click", (event) => {
    track.mute = !track.mute;
    event.currentTarget.classList.toggle("active", track.mute);
    applyMuteSolo();
  });
  row.querySelector(".solo").addEventListener("click", (event) => {
    track.solo = !track.solo;
    event.currentTarget.classList.toggle("active", track.solo);
    applyMuteSolo();
  });
  row.querySelector(".volume").addEventListener("input", (event) => setTrackGain(track, Number(event.target.value), row, track.channel));
  row.querySelector(".pan").addEventListener("input", (event) => {
    track.pan = Number(event.target.value);
    const value = Math.abs(track.pan) < .02 ? "C" : track.pan < 0 ? `L${Math.round(-track.pan * 100)}` : `R${Math.round(track.pan * 100)}`;
    row.querySelector(".pan + output").textContent = value;
    if (track.panNode) track.panNode.pan.value = track.pan;
  });
  row.querySelector(".track-wave").addEventListener("click", (event) => {
    if (track.id === "heartbeat" && event.detail > 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    seek((event.clientX - rect.left) / state.zoom);
  });
}

function setTrackGain(track, db, row, channel) {
  track.gainDb = db;
  if (row) {
    row.querySelector(".volume").value = db;
    row.querySelector(".volume + output").textContent = db.toFixed(1);
  }
  if (channel) {
    channel.querySelector("input").value = db;
    channel.querySelector("small").textContent = `${db.toFixed(1)} dB`;
  }
  applyMuteSolo();
}

function applyMuteSolo() {
  const anySolo = [...state.tracks.values()].some((track) => track.solo);
  for (const track of state.tracks.values()) {
    const audible = !track.mute && (!anySolo || track.solo);
    if (track.gainNode) track.gainNode.gain.setTargetAtTime(audible ? dbToGain(track.gainDb) : 0, state.audioContext.currentTime, .01);
  }
}

function bindPhaseDrag(element) {
  let dragging = false;
  let originX = 0;
  let originPhase = 0;
  element.addEventListener("pointerdown", (event) => {
    dragging = true;
    originX = event.clientX;
    originPhase = Number($("phase").value);
    element.setPointerCapture(event.pointerId);
  });
  element.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const phase = snapTime(originPhase + (event.clientX - originX) / state.zoom);
    $("phase").value = Math.min(Number($("phase").max), phase).toFixed(3);
    $("phaseValue").textContent = `${Number($("phase").value).toFixed(3)} s`;
    state.phaseTouched = true;
    markDirty("Heartbeat clip moved — render to apply high-quality phase alignment.");
    drawTrack(state.tracks.get("heartbeat"));
  });
  element.addEventListener("pointerup", () => { dragging = false; });
}

async function togglePlayback() {
  if (!state.tracks.size) return;
  await ensureAudioContext();
  if (state.playing) pausePlayback(); else playFrom(state.pausedAt);
}

function playFrom(position) {
  stopSources();
  if (position >= state.duration) position = 0;
  const when = state.audioContext.currentTime + .03;
  for (const track of state.tracks.values()) {
    if (!track.buffer || position >= track.buffer.duration) continue;
    const source = state.audioContext.createBufferSource();
    const gain = state.audioContext.createGain();
    const pan = state.audioContext.createStereoPanner();
    const analyser = state.audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.buffer = track.buffer;
    source.connect(gain).connect(pan).connect(analyser).connect(state.masterGain);
    track.gainNode = gain;
    track.panNode = pan;
    track.analyser = analyser;
    pan.pan.value = track.pan;
    source.start(when, position);
    state.sources.push(source);
  }
  state.pausedAt = position;
  state.startedAt = when - position;
  state.playing = true;
  $("play").textContent = "❚❚";
  $("play").classList.add("playing");
  applyMuteSolo();
  animate();
}

function pausePlayback() {
  state.pausedAt = currentPosition();
  stopSources();
  state.playing = false;
  $("play").textContent = "▶";
  $("play").classList.remove("playing");
  cancelAnimationFrame(state.animationFrame);
  updatePlayhead();
}

function stopPlayback() {
  stopSources();
  state.playing = false;
  state.pausedAt = 0;
  $("play").textContent = "▶";
  $("play").classList.remove("playing");
  cancelAnimationFrame(state.animationFrame);
  updatePlayhead();
}

function stopSources() {
  for (const source of state.sources) { try { source.stop(); } catch (_) {} }
  state.sources = [];
}

function seek(position) {
  const next = Math.max(0, Math.min(state.duration, position));
  const wasPlaying = state.playing;
  if (wasPlaying) playFrom(next); else { state.pausedAt = next; updatePlayhead(); }
}

function animate() {
  updatePlayhead();
  updateMeters();
  if (currentPosition() >= state.duration) { stopPlayback(); return; }
  state.animationFrame = requestAnimationFrame(animate);
}

function updatePlayhead() {
  const position = currentPosition();
  $("timeNow").textContent = formatTime(position);
  const bpm = Number($("bpm").value) || 120;
  const meter = Number($("meter").value) || 4;
  const beatIndex = Math.max(0, Math.floor((position - Number($("phase").value)) / (60 / bpm)));
  $("barBeat").textContent = `BAR ${Math.floor(beatIndex / meter) + 1} · BEAT ${(beatIndex % meter) + 1}`;
  $("playhead").style.left = `calc(var(--track-head) + ${position * state.zoom}px)`;
}

function updateMeters() {
  const data = new Uint8Array(128);
  for (const track of state.tracks.values()) {
    if (!track.analyser || !track.channel) continue;
    track.analyser.getByteTimeDomainData(data);
    let energy = 0;
    for (const value of data) energy += Math.pow((value - 128) / 128, 2);
    const rms = Math.sqrt(energy / data.length);
    track.channel.querySelector(".meter span").style.height = `${Math.min(100, rms * 230)}%`;
  }
}

async function renderSession() {
  if (!state.jobId) return;
  $("render").disabled = true;
  $("progress").hidden = false;
  setStatus("Rendering heartbeat stretch, phase alignment, limiter, WAV and MP3…");
  try {
    const response = await fetch(`/api/jobs/${state.jobId}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bpm: Number($("bpm").value),
        first_beat_seconds: Number($("phase").value),
        heartbeat_gain_db: Number($("heartbeatGain").value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Render failed");
    const heartbeat = state.tracks.get("heartbeat");
    heartbeat.url = `${payload.heartbeat_layer_url}?revision=${payload.revision}`;
    await loadTrackBuffer(heartbeat);
    updateDownloads(payload.final_mix_wav_url, payload.final_mix_mp3_url);
    markRendered();
    drawAll();
    setStatus(`Revision ${payload.revision} rendered successfully.`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    $("progress").hidden = true;
    $("render").disabled = false;
  }
}

function updateDownloads(wav, mp3) {
  const wavLink = $("downloadWav");
  wavLink.href = wav || "#";
  wavLink.classList.toggle("disabled", !wav);
  const mp3Link = $("downloadMp3");
  mp3Link.href = mp3 || "#";
  mp3Link.classList.toggle("disabled", !mp3);
}

function updateAnalysisPanel(payload) {
  const analysis = payload.analysis;
  const stability = analysis.tempo_stability || {};
  const loop = payload.heartbeat_summary.best_loop;
  $("confidence").textContent = `${Math.round(analysis.beat_tracking_confidence * 100)}%`;
  $("gridError").textContent = stability.grid_error_p95_seconds == null ? "n/a" : `${(stability.grid_error_p95_seconds * 1000).toFixed(1)} ms`;
  $("heartBpm").textContent = `${Number(loop.local_bpm).toFixed(1)}`;
  $("loopDuration").textContent = `${Number(loop.duration_seconds).toFixed(3)} s`;
  $("warnings").innerHTML = (analysis.warnings || []).map((warning) => `<div class="warning-item">${escapeHtml(warning)}</div>`).join("");
}

function adjustZoom(delta) {
  const input = $("zoom");
  input.value = Math.max(Number(input.min), Math.min(Number(input.max), Number(input.value) + delta));
  state.zoom = Number(input.value);
  drawAll();
}

function drawAll() {
  if (!state.duration) return;
  const available = Math.max(650, $("timelineScroll").clientWidth - 164);
  const width = Math.max(available, state.duration * state.zoom);
  $("timelineContent").style.width = `${width + 164}px`;
  drawRuler(width);
  for (const track of state.tracks.values()) drawTrack(track, width);
  updatePlayhead();
}

function setupCanvas(canvas, cssWidth, cssHeight) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const internalWidth = Math.min(32760, Math.max(1, Math.floor(cssWidth * dpr)));
  canvas.width = internalWidth;
  canvas.height = Math.floor(cssHeight * dpr);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  return { ctx: canvas.getContext("2d"), width: internalWidth, height: canvas.height, scale: internalWidth / cssWidth };
}

function drawRuler(forcedWidth) {
  const canvas = $("ruler");
  const cssWidth = forcedWidth || Math.max(650, state.duration * state.zoom);
  const { ctx, width, height, scale } = setupCanvas(canvas, cssWidth, 28);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#15191e";
  ctx.fillRect(0, 0, width, height);
  const bpm = Number($("bpm").value) || 120;
  const meter = Number($("meter").value) || 4;
  const period = 60 / bpm;
  const phase = Number($("phase").value) || 0;
  ctx.font = `${9 * scale}px Segoe UI`;
  for (let time = phase, beat = 0; time <= state.duration + period; time += period, beat += 1) {
    const x = time * state.zoom * scale;
    const downbeat = beat % meter === 0;
    ctx.strokeStyle = downbeat ? "#68717d" : "#343b45";
    ctx.beginPath(); ctx.moveTo(x, downbeat ? 0 : height * .45); ctx.lineTo(x, height); ctx.stroke();
    if (downbeat) { ctx.fillStyle = "#8b949f"; ctx.fillText(String(Math.floor(beat / meter) + 1), x + 4 * scale, 10 * scale); }
  }
}

function drawTrack(track, forcedWidth) {
  if (!track?.row || !track.buffer) return;
  const canvas = track.row.querySelector("canvas");
  const cssWidth = forcedWidth || Math.max(650, state.duration * state.zoom);
  const { ctx, width, height, scale } = setupCanvas(canvas, cssWidth, 115);
  ctx.clearRect(0, 0, width, height);
  drawGrid(ctx, width, height, scale);
  const data = track.buffer.getChannelData(0);
  const visualShift = track.id === "heartbeat" ? (Number($("phase").value) - state.renderedPhase) * state.zoom * scale : 0;
  const samplesPerPixel = Math.max(1, Math.floor(data.length / width));
  ctx.strokeStyle = track.color;
  ctx.globalAlpha = track.mute ? .34 : .9;
  ctx.lineWidth = Math.max(1, scale);
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const start = Math.floor(x * samplesPerPixel);
    const end = Math.min(data.length, start + samplesPerPixel);
    let min = 1, max = -1;
    for (let i = start; i < end; i++) { const value = data[i]; if (value < min) min = value; if (value > max) max = value; }
    const drawX = x + visualShift;
    if (drawX < 0 || drawX > width) continue;
    ctx.moveTo(drawX, (1 + min) * .5 * height);
    ctx.lineTo(drawX, (1 + max) * .5 * height);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawGrid(ctx, width, height, scale) {
  const bpm = Number($("bpm").value) || 120;
  const meter = Number($("meter").value) || 4;
  const period = 60 / bpm;
  const phase = Number($("phase").value) || 0;
  for (let time = phase, beat = 0; time <= state.duration + period; time += period, beat += 1) {
    const x = time * state.zoom * scale;
    ctx.strokeStyle = beat % meter === 0 ? "#343d47" : "#222930";
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
  }
  ctx.strokeStyle = "#272e36";
  ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}
