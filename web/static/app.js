/**
 * UA-DETRAC Intelligent Traffic Tracker - Frontend Client
 * Handles API initialization, WebSocket streaming, canvas rendering, and metrics.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const videoSelect = document.getElementById('video-select');
    const modelSelect = document.getElementById('model-select');
    const trackerSelect = document.getElementById('tracker-select');

    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const maxAgeSlider = document.getElementById('max-age-slider');
    const maxAgeVal = document.getElementById('max-age-val');
    const trailsToggle = document.getElementById('trails-toggle');

    const btnStart = document.getElementById('btn-start');
    const btnPause = document.getElementById('btn-pause');
    const btnStop = document.getElementById('btn-stop');

    const canvas = document.getElementById('stream-canvas');
    const ctx = canvas.getContext('2d');
    const canvasPlaceholder = document.getElementById('canvas-placeholder');
    const canvasHud = document.getElementById('canvas-hud');
    const hudTracker = document.getElementById('hud-tracker');
    const hudModel = document.getElementById('hud-model');
    const hudRes = document.getElementById('hud-res');

    const statusPill = document.getElementById('status-pill');
    const statusText = document.getElementById('status-text');

    const statFps = document.getElementById('stat-fps');
    const statActive = document.getElementById('stat-active');
    const statTotal = document.getElementById('stat-total');
    const statFrame = document.getElementById('stat-frame');
    const statTotalFrames = document.getElementById('stat-total-frames');

    const countCar = document.getElementById('count-car');
    const countBus = document.getElementById('count-bus');
    const countVan = document.getElementById('count-van');
    const countTruck = document.getElementById('count-truck');

    // State Variables
    let ws = null;
    let isStreaming = false;
    let isPaused = false;
    let currentImage = new Image();

    // 1. Initial Setup: Slider dynamic labels
    confSlider.addEventListener('input', (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
        sendConfigUpdate();
    });

    maxAgeSlider.addEventListener('input', (e) => {
        maxAgeVal.textContent = e.target.value;
        sendConfigUpdate();
    });

    trailsToggle.addEventListener('change', () => {
        sendConfigUpdate();
    });

    // 2. Fetch API Data
    async function loadInitialOptions() {
        try {
            // Load Videos
            const resVideos = await fetch('/api/videos');
            const videos = await resVideos.json();
            videoSelect.innerHTML = '';
            videos.forEach((vid, idx) => {
                const opt = document.createElement('option');
                opt.value = vid.id;
                opt.textContent = vid.name;
                if (idx === 0) opt.selected = true;
                videoSelect.appendChild(opt);
            });

            // Load Models
            const resModels = await fetch('/api/models');
            const models = await resModels.json();
            modelSelect.innerHTML = '';
            models.forEach((m) => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.name;
                if (m.is_default) opt.selected = true;
                modelSelect.appendChild(opt);
            });

            // Load Trackers
            const resTrackers = await fetch('/api/trackers');
            const trackers = await resTrackers.json();
            trackerSelect.innerHTML = '';
            trackers.forEach((t, idx) => {
                const opt = document.createElement('option');
                opt.value = t.id;
                opt.textContent = t.name;
                if (idx === 0) opt.selected = true;
                trackerSelect.appendChild(opt);
            });

            updateStatus('ready', 'Connected (Ready)');
        } catch (err) {
            console.error('Failed to load initial options:', err);
            updateStatus('error', 'API Load Error');
        }
    }

    // 3. UI Status updates
    function updateStatus(state, label) {
        statusText.textContent = label;
        statusPill.className = 'status-pill';
        if (state === 'connected' || state === 'ready') {
            statusPill.classList.add('connected');
        } else if (state === 'streaming') {
            statusPill.classList.add('streaming');
        }
    }

    // 4. WebSocket Management
    function setupWebSocket() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('WebSocket connected to stream service');
            updateStatus('connected', 'Connected (Ready)');
        };

        ws.onclose = () => {
            console.log('WebSocket connection closed');
            if (isStreaming) {
                stopStreamUI();
            }
            updateStatus('disconnected', 'Disconnected');
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            updateStatus('error', 'Connection Error');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerMessage(data);
            } catch (e) {
                console.error('Failed to parse WS message:', e);
            }
        };
    }

    function handleServerMessage(data) {
        if (data.type === 'frame') {
            // Render Frame onto Canvas
            currentImage.onload = () => {
                if (canvas.width !== currentImage.width || canvas.height !== currentImage.height) {
                    canvas.width = currentImage.width;
                    canvas.height = currentImage.height;
                    hudRes.textContent = `${currentImage.width}x${currentImage.height}`;
                }
                ctx.drawImage(currentImage, 0, 0);
            };
            currentImage.src = 'data:image/jpeg;base64,' + data.frame;

            // Update Telemetry & Statistics
            statFps.textContent = (data.fps || 0).toFixed(1);
            statActive.textContent = data.active_tracks || 0;
            statTotal.textContent = data.total_tracks || 0;
            statFrame.textContent = data.frame_idx || 0;

            if (data.total_frames) {
                statTotalFrames.textContent = `of ${data.total_frames} frames`;
            }

            // Update Class Breakdown
            if (data.classes) {
                countCar.textContent = data.classes.car || 0;
                countBus.textContent = data.classes.bus || 0;
                countVan.textContent = data.classes.van || 0;
                countTruck.textContent = data.classes.truck || 0;
            }
        } else if (data.type === 'status') {
            if (data.status === 'completed') {
                stopStreamUI();
                updateStatus('connected', 'Completed');
            } else if (data.status === 'error') {
                alert(`Tracking error: ${data.message}`);
                stopStreamUI();
                updateStatus('error', 'Error');
            }
        }
    }

    function sendConfigUpdate() {
        if (!ws || ws.readyState !== WebSocket.OPEN || !isStreaming) return;
        ws.send(JSON.stringify({
            action: 'update_config',
            conf_thresh: parseFloat(confSlider.value),
            max_age: parseInt(maxAgeSlider.value, 10),
            show_trails: trailsToggle.checked
        }));
    }

    // 5. Stream Control Handlers
    function startStream() {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            setupWebSocket();
            // Retry after brief connection wait
            setTimeout(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    executeStart();
                } else {
                    alert('Cannot connect to WebSocket stream server.');
                }
            }, 500);
            return;
        }
        executeStart();
    }

    function executeStart() {
        const payload = {
            action: 'start',
            video: videoSelect.value,
            model: modelSelect.value,
            tracker: trackerSelect.value,
            conf_thresh: parseFloat(confSlider.value),
            max_age: parseInt(maxAgeSlider.value, 10),
            show_trails: trailsToggle.checked
        };

        ws.send(JSON.stringify(payload));

        isStreaming = true;
        isPaused = false;

        // UI Updates
        btnStart.disabled = true;
        btnPause.disabled = false;
        btnPause.innerHTML = '<span>⏸ Pause</span>';
        btnStop.disabled = false;
        videoSelect.disabled = true;
        modelSelect.disabled = true;
        trackerSelect.disabled = true;

        canvasPlaceholder.style.display = 'none';
        canvasHud.style.display = 'flex';
        hudTracker.textContent = trackerSelect.options[trackerSelect.selectedIndex].text.split(' ')[0];
        hudModel.textContent = modelSelect.options[modelSelect.selectedIndex].text.split(' ')[0];

        updateStatus('streaming', 'Streaming Live');
    }

    function togglePause() {
        if (!ws || ws.readyState !== WebSocket.OPEN || !isStreaming) return;

        if (isPaused) {
            ws.send(JSON.stringify({ action: 'resume' }));
            isPaused = false;
            btnPause.innerHTML = '<span>⏸ Pause</span>';
            updateStatus('streaming', 'Streaming Live');
        } else {
            ws.send(JSON.stringify({ action: 'pause' }));
            isPaused = true;
            btnPause.innerHTML = '<span>▶ Resume</span>';
            updateStatus('connected', 'Stream Paused');
        }
    }

    function stopStream() {
        if (ws && ws.readyState === WebSocket.OPEN && isStreaming) {
            ws.send(JSON.stringify({ action: 'stop' }));
        }
        stopStreamUI();
    }

    function stopStreamUI() {
        isStreaming = false;
        isPaused = false;

        btnStart.disabled = false;
        btnPause.disabled = true;
        btnPause.innerHTML = '<span>⏸ Pause</span>';
        btnStop.disabled = true;
        videoSelect.disabled = false;
        modelSelect.disabled = false;
        trackerSelect.disabled = false;

        updateStatus('connected', 'Connected (Idle)');
    }

    // Bind Button Listeners
    btnStart.addEventListener('click', startStream);
    btnPause.addEventListener('click', togglePause);
    btnStop.addEventListener('click', stopStream);

    // Initial load
    loadInitialOptions();
    setupWebSocket();
});
