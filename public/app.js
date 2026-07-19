// Interactive Logic for AgriSense Dashboard

document.addEventListener("DOMContentLoaded", () => {
    // API Endpoint root (relative path for Vercel, points to localhost:5000 in local dev)
    const API_ROOT = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" ? "http://localhost:5000" : "";

    // State Variables
    let activeState = {
        current_scene_path: "",
        is_liss4: false,
        preprocessed: false,
        annotations_loaded: false,
        trained_model_type: "",
        active_field_metrics: null
    };

    // Chart instances
    let overviewChartInstance = null;
    let trainingChartInstance = null;

    // Initialize UI Elements
    initTabs();
    initSliders();
    initThemeSwitcher();
    initEventListeners();
    
    // Check initial info from API
    fetchPipelineInfo();

    // ==========================================
    // INITIALIZATION & TAB CONTROLS
    // ==========================================
    function initTabs() {
        // Main tabs
        const tabButtons = document.querySelectorAll(".nav-tabs .tab-btn");
        const tabPanels = document.querySelectorAll(".tab-content-container .tab-panel");

        tabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const target = btn.getAttribute("data-tab");

                tabButtons.forEach(b => b.classList.remove("active"));
                tabPanels.forEach(p => p.classList.remove("active"));

                btn.classList.add("active");
                const targetPanel = document.getElementById(target);
                if (targetPanel) {
                    targetPanel.classList.add("active");
                }

                // If switching to overview, reload metrics/charts
                if (target === "tab-overview") {
                    fetchPipelineInfo();
                }
            });
        });

        // Subtabs (Annotation page)
        const subTabButtons = document.querySelectorAll(".sub-tabs .sub-tab-btn");
        const subTabPanels = document.querySelectorAll(".tab-panel .sub-tab-panel");

        subTabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const target = btn.getAttribute("data-subtab");

                subTabButtons.forEach(b => b.classList.remove("active"));
                subTabPanels.forEach(p => p.classList.remove("active"));

                btn.classList.add("active");
                const targetPanel = document.getElementById(target);
                if (targetPanel) {
                    targetPanel.classList.add("active");
                }
            });
        });
    }

    function initSliders() {
        // Cloud Cover Slider
        const cloudSlider = document.getElementById("cloud-slider");
        const cloudVal = document.getElementById("cloud-val");
        if (cloudSlider && cloudVal) {
            cloudSlider.addEventListener("input", (e) => {
                cloudVal.textContent = e.target.value;
            });
        }

        // Synth Fields Slider
        const synthFields = document.getElementById("synth-fields");
        const synthFieldsVal = document.getElementById("synth-fields-val");
        if (synthFields && synthFieldsVal) {
            synthFields.addEventListener("input", (e) => {
                synthFieldsVal.textContent = e.target.value;
            });
        }

        // Epochs slider
        const epochsSlider = document.getElementById("train-epochs");
        const epochsVal = document.getElementById("train-epochs-val");
        if (epochsSlider && epochsVal) {
            epochsSlider.addEventListener("input", (e) => {
                epochsVal.textContent = e.target.value;
            });
        }
    }

    function initThemeSwitcher() {
        const darkBtn = document.getElementById("theme-dark");
        const lightBtn = document.getElementById("theme-light");

        if (darkBtn && lightBtn) {
            darkBtn.addEventListener("click", () => {
                document.body.classList.remove("light-theme");
                darkBtn.classList.add("active");
                lightBtn.classList.remove("active");
            });

            lightBtn.addEventListener("click", () => {
                document.body.classList.add("light-theme");
                lightBtn.classList.add("active");
                darkBtn.classList.remove("active");
            });
        }
    }

    // ==========================================
    // API CALLS & CORE LOGIC
    // ==========================================
    function fetchPipelineInfo() {
        fetch(`${API_ROOT}/api/info`)
            .then(res => res.json())
            .then(data => {
                activeState = data;
                updatePipelineTrackers();
                renderOverviewPanel();
            })
            .catch(err => console.error("Error fetching pipeline info:", err));
    }

    function updatePipelineTrackers() {
        // 1. Ingestion
        const stepIng = document.getElementById("step-ingestion");
        const labelIng = stepIng.querySelector(".step-status");
        if (activeState.current_scene_path) {
            stepIng.classList.add("completed");
            labelIng.className = "step-status status-healthy";
            labelIng.textContent = "Active";
        } else {
            stepIng.classList.remove("completed");
            labelIng.className = "step-status status-pending";
            labelIng.textContent = "Pending";
        }

        // 2. Preprocessing
        const stepPrep = document.getElementById("step-preprocess");
        const labelPrep = stepPrep.querySelector(".step-status");
        if (activeState.preprocessed) {
            stepPrep.classList.add("completed");
            labelPrep.className = "step-status status-healthy";
            labelPrep.textContent = "Completed";
        } else {
            stepPrep.classList.remove("completed");
            labelPrep.className = "step-status status-pending";
            labelPrep.textContent = "Pending";
        }

        // 3. Annotation
        const stepAnn = document.getElementById("step-annotation");
        const labelAnn = stepAnn.querySelector(".step-status");
        if (activeState.annotations_loaded) {
            stepAnn.classList.add("completed");
            labelAnn.className = "step-status status-healthy";
            labelAnn.textContent = "Loaded";
        } else {
            stepAnn.classList.remove("completed");
            labelAnn.className = "step-status status-pending";
            labelAnn.textContent = "Pending";
        }

        // 4. Augmentation
        const stepAug = document.getElementById("step-augmentation");
        const labelAug = stepAug.querySelector(".step-status");
        if (activeState.annotations_loaded) {
            stepAug.classList.add("completed");
            labelAug.className = "step-status status-healthy";
            labelAug.textContent = "Ready";
        } else {
            stepAug.classList.remove("completed");
            labelAug.className = "step-status status-pending";
            labelAug.textContent = "Pending";
        }

        // 5. Training
        const stepTrain = document.getElementById("step-training");
        const labelTrain = stepTrain.querySelector(".step-status");
        if (activeState.trained_model_type) {
            stepTrain.classList.add("completed");
            labelTrain.className = "step-status status-healthy";
            labelTrain.textContent = activeState.trained_model_type.toUpperCase();
        } else {
            stepTrain.classList.remove("completed");
            labelTrain.className = "step-status status-pending";
            labelTrain.textContent = "Pending";
        }

        // 6. Predictions
        const stepPred = document.getElementById("step-predictions");
        const labelPred = stepPred.querySelector(".step-status");
        if (activeState.active_field_metrics) {
            stepPred.classList.add("completed");
            labelPred.className = "step-status status-healthy";
            labelPred.textContent = "Generated";
        } else {
            stepPred.classList.remove("completed");
            labelPred.className = "step-status status-pending";
            labelPred.textContent = "Pending";
        }
    }

    function renderOverviewPanel() {
        const previewContainer = document.getElementById("overview-preview-container");
        const mapElement = document.getElementById("simulated-map-element");
        const imgElement = document.getElementById("overview-scene-image");
        
        // Show actual image if preprocessed fcc exists, otherwise show placeholder simulated map
        if (activeState.preprocessed) {
            mapElement.classList.add("hidden");
            imgElement.classList.remove("hidden");
            // Hit preprocessor FCC endpoint
            fetch(`${API_ROOT}/api/preprocess`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    apply_clahe: true,
                    filter_opt: "Bilateral Filter (Edge Preserving)",
                    display_mode: "False Color Composite (FCC: NIR-Red-Green)"
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === "success") {
                    imgElement.src = `data:image/png;base64,${data.image}`;
                }
            });
        } else {
            mapElement.classList.remove("hidden");
            imgElement.classList.add("hidden");
        }

        // Update metric values
        const metricsRow = document.getElementById("overview-metrics-row");
        if (activeState.active_field_metrics && activeState.active_field_metrics.length > 0) {
            const metrics = activeState.active_field_metrics;
            const fieldsCount = metrics.length;
            const totalArea = metrics.reduce((acc, f) => acc + f.area_hectares, 0);
            const totalYield = metrics.reduce((acc, f) => acc + f.yield_tons, 0);

            document.getElementById("metric-fields-val").textContent = fieldsCount;
            document.getElementById("metric-area-val").textContent = `${totalArea.toFixed(1)} ha`;
            document.getElementById("metric-yield-val").textContent = `${totalYield.toFixed(1)} t`;

            // Draw Overview Crop Yield Chart
            const crops = {};
            metrics.forEach(f => {
                const type = f.crop_type.charAt(0).toUpperCase() + f.crop_type.slice(1);
                crops[type] = (crops[type] || 0) + f.yield_tons;
            });
            drawOverviewChart(Object.keys(crops), Object.values(crops));
        } else {
            // Default simulated stats
            document.getElementById("metric-fields-val").textContent = "16";
            document.getElementById("metric-area-val").textContent = "104.5 ha";
            document.getElementById("metric-yield-val").textContent = "4.5 t/ha";

            // Default simulated crop chart
            drawOverviewChart(
                ["Rice", "Sugarcane", "Cotton", "Wheat", "Maize"],
                [4.2, 75.0, 2.1, 3.9, 5.2],
                "Expected Yield (Tons/ha)"
            );
        }
    }

    function drawOverviewChart(labels, data, label = "Yield by Crop (Tons)") {
        const ctx = document.getElementById("overview-chart").getContext("2d");
        
        if (overviewChartInstance) {
            overviewChartInstance.destroy();
        }

        overviewChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: 'rgba(0, 255, 183, 0.45)',
                    borderColor: '#00ffb7',
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: 'var(--text-color)' }
                    }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'var(--text-muted)' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'var(--text-muted)' }
                    }
                }
            }
        });
    }

    // ==========================================
    // EVENT LISTENERS & ACTION HANDLERS
    // ==========================================
    function initEventListeners() {
        
        // --- TAB 1: ACQUISITION ---
        const btnSearch = document.getElementById("btn-catalog-search");
        btnSearch.addEventListener("click", () => {
            btnSearch.disabled = true;
            btnSearch.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Querying...';
            
            const satellite = document.getElementById("satellite-type").value;
            const sensor = document.getElementById("sensor-type").value;
            const startDate = document.getElementById("start-date").value;
            const endDate = document.getElementById("end-date").value;
            const username = document.getElementById("bhoonidhi-username").value;
            const password = document.getElementById("bhoonidhi-password").value;

            fetch(`${API_ROOT}/api/acquire`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: "search",
                    satellite: satellite,
                    sensor: sensor,
                    start_date: startDate,
                    end_date: endDate,
                    username: username,
                    password: password
                })
            })
            .then(res => res.json())
            .then(data => {
                btnSearch.disabled = false;
                btnSearch.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Search Bhoonidhi Catalog';
                
                const resultsContainer = document.getElementById("catalog-results-container");
                if (data.status === "success" && data.results.length > 0) {
                    let tableHtml = `
                        <table class="bhoonidhi-table">
                            <thead>
                                <tr>
                                    <th>Product ID</th>
                                    <th>Satellite</th>
                                    <th>Sensor</th>
                                    <th>Clouds %</th>
                                    <th>Date</th>
                                    <th>Action</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;
                    data.results.forEach(scene => {
                        tableHtml += `
                            <tr>
                                <td>${scene.product_id}</td>
                                <td>${scene.satellite}</td>
                                <td>${scene.sensor}</td>
                                <td>${scene.cloud_cover}</td>
                                <td>${scene.acquisition_date}</td>
                                <td><button class="btn-sm btn-download-scene" data-product="${scene.product_id}" data-sensor="${scene.sensor}">Load</button></td>
                            </tr>
                        `;
                    });
                    tableHtml += `</tbody></table>`;
                    resultsContainer.innerHTML = tableHtml;
                    
                    // Attach load event
                    document.querySelectorAll(".btn-download-scene").forEach(btn => {
                        btn.addEventListener("click", (e) => {
                            const prodId = btn.getAttribute("data-product");
                            const sensorVal = btn.getAttribute("data-sensor");
                            loadBhoonidhiScene(prodId, sensorVal, btn);
                        });
                    });
                } else {
                    resultsContainer.innerHTML = '<p class="info-text">No scenes found matching the criteria.</p>';
                }
            })
            .catch(err => {
                btnSearch.disabled = false;
                btnSearch.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Search Bhoonidhi Catalog';
                console.error("Search failed:", err);
            });
        });

        // Quick Load buttons
        document.getElementById("btn-load-liss3").addEventListener("click", () => {
            loadSampleDataset("LISS-3");
        });
        document.getElementById("btn-load-liss4").addEventListener("click", () => {
            loadSampleDataset("LISS-4");
        });

        // --- TAB 2: PREPROCESSING ---
        document.getElementById("btn-preprocess").addEventListener("click", () => {
            const applyClahe = document.getElementById("apply-clahe").checked;
            const filterOpt = document.getElementById("filter-select").value;
            const displayMode = document.querySelector('input[name="display-mode"]:checked').value;
            const selectedBand = document.getElementById("band-select").value;

            const imgPreview = document.getElementById("preprocess-img");
            const loader = document.getElementById("preprocess-loader");
            const captionEl = document.getElementById("preprocess-caption");

            loader.classList.remove("hidden");
            imgPreview.src = "";
            captionEl.textContent = "Processing image...";

            fetch(`${API_ROOT}/api/preprocess`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    apply_clahe: applyClahe,
                    filter_opt: filterOpt,
                    display_mode: displayMode,
                    selected_band: selectedBand
                })
            })
            .then(res => res.json())
            .then(data => {
                loader.classList.add("hidden");
                if (data.status === "success") {
                    imgPreview.src = `data:image/png;base64,${data.image}`;
                    captionEl.textContent = data.caption;
                    fetchPipelineInfo(); // Sync trackers
                } else {
                    captionEl.textContent = "Error running preprocessing.";
                }
            })
            .catch(err => {
                loader.classList.add("hidden");
                captionEl.textContent = "Request failed.";
                console.error(err);
            });
        });

        // Hide/show band select container based on display mode
        document.querySelectorAll('input[name="display-mode"]').forEach(radio => {
            radio.addEventListener("change", (e) => {
                const bandSelectContainer = document.getElementById("band-select-container");
                if (e.target.value === "Individual Band (Gray Scale)") {
                    bandSelectContainer.classList.remove("hidden");
                } else {
                    bandSelectContainer.classList.add("hidden");
                }
            });
        });

        // --- TAB 3: ANNOTATION & AUGMENT ---
        // VIA Demarcation
        const btnGenVia = document.getElementById("btn-generate-via");
        btnGenVia.addEventListener("click", () => {
            const imgPreview = document.getElementById("via-img");
            const loader = document.getElementById("via-loader");
            const captionEl = document.getElementById("via-caption");

            loader.classList.remove("hidden");
            imgPreview.src = "";
            captionEl.textContent = "Generating boundary labels...";

            fetch(`${API_ROOT}/api/annotate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: "generate_mock" })
            })
            .then(res => res.json())
            .then(data => {
                loader.classList.add("hidden");
                if (data.status === "success" || data.status === "warning") {
                    imgPreview.src = `data:image/png;base64,${data.image}`;
                    captionEl.textContent = data.message;
                    fetchPipelineInfo();
                } else {
                    captionEl.textContent = "Error parsing annotations.";
                }
            })
            .catch(err => {
                loader.classList.add("hidden");
                captionEl.textContent = "Request failed.";
                console.error(err);
            });
        });

        // File upload manual parser
        const fileUploadInput = document.getElementById("via-file-upload");
        const fileUploadName = document.getElementById("file-upload-name");
        fileUploadInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (!file) return;

            fileUploadName.textContent = file.name;
            const reader = new FileReader();
            reader.onload = function(event) {
                const base64Content = event.target.result.split(',')[1];
                
                const imgPreview = document.getElementById("via-img");
                const loader = document.getElementById("via-loader");
                const captionEl = document.getElementById("via-caption");

                loader.classList.remove("hidden");
                imgPreview.src = "";
                
                fetch(`${API_ROOT}/api/annotate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        action: "upload",
                        file_content: base64Content
                    })
                })
                .then(res => res.json())
                .then(data => {
                    loader.classList.add("hidden");
                    if (data.status === "success") {
                        imgPreview.src = `data:image/png;base64,${data.image}`;
                        captionEl.textContent = data.message;
                        fetchPipelineInfo();
                    } else {
                        captionEl.textContent = "Failed to parse uploaded JSON.";
                    }
                })
                .catch(err => {
                    loader.classList.add("hidden");
                    captionEl.textContent = "Upload failed.";
                    console.error(err);
                });
            };
            reader.readAsDataURL(file);
        });

        // Albumentations Coordinated Augmentation
        document.getElementById("btn-run-augment").addEventListener("click", () => {
            const btn = document.getElementById("btn-run-augment");
            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Transforming...';

            const rot = document.getElementById("aug-rot").checked;
            const flip = document.getElementById("aug-flip").checked;
            const noise = document.getElementById("aug-noise").checked;

            fetch(`${API_ROOT}/api/augment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rot, flip, noise })
            })
            .then(res => res.json())
            .then(data => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-play"></i> Generate Augmented Sample';

                if (data.status === "success") {
                    document.getElementById("aug-img-fcc").src = `data:image/png;base64,${data.image}`;
                    document.getElementById("aug-img-mask").src = `data:image/png;base64,${data.mask}`;
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-play"></i> Generate Augmented Sample';
                console.error(err);
            });
        });

        // Generative AI Synth
        document.getElementById("btn-run-synth").addEventListener("click", () => {
            const btn = document.getElementById("btn-run-synth");
            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Synthesizing...';

            const fieldsCount = parseInt(document.getElementById("synth-fields").value);
            const size = parseInt(document.getElementById("synth-size").value);
            const seed = parseInt(document.getElementById("synth-seed").value);

            fetch(`${API_ROOT}/api/synthesize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fields_count: fieldsCount, size, seed })
            })
            .then(res => res.json())
            .then(data => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-flask"></i> Synthesize Data with Generative AI';

                if (data.status === "success") {
                    document.getElementById("synth-img-fcc").src = `data:image/png;base64,${data.image}`;
                    document.getElementById("synth-img-mask").src = `data:image/png;base64,${data.mask}`;
                    document.getElementById("synth-message").textContent = `Successfully generated ${data.fields_count} crop fields with matching semantic labels (seed: ${seed}).`;
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-flask"></i> Synthesize Data with Generative AI';
                console.error(err);
            });
        });

        // --- TAB 4: MODEL TRAINING ---
        document.getElementById("btn-train").addEventListener("click", () => {
            const btn = document.getElementById("btn-train");
            const loader = document.getElementById("training-loader");
            const resultsPanel = document.getElementById("training-results-panel");
            const chartContainer = document.getElementById("loss-chart-container");

            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Training...';
            loader.classList.remove("hidden");
            chartContainer.style.display = "none";
            resultsPanel.innerHTML = "<p>Training model in serverless sandbox...</p>";

            const paradigm = document.querySelector('input[name="train-paradigm"]:checked').value;
            const epochs = parseInt(document.getElementById("train-epochs").value);

            fetch(`${API_ROOT}/api/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paradigm, epochs })
            })
            .then(res => res.json())
            .then(data => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-graduation-cap"></i> Start Model Training';
                loader.classList.add("hidden");

                if (data.status === "success") {
                    let resultsHtml = `<h5>Training Complete!</h5><p>Paradigm: <strong>${data.model_type.toUpperCase()}</strong></p>`;
                    
                    if (data.model_type === "unsupervised") {
                        resultsHtml += `<p>Cluster class mappings identified from spectral reflections:</p><ul>`;
                        data.mapping.forEach(item => {
                            resultsHtml += `<li>Cluster ${item[0]}: <strong>${item[1]}</strong></li>`;
                        });
                        resultsHtml += `</ul>`;
                    } else {
                        resultsHtml += `<p>${data.message}</p>`;
                    }

                    resultsPanel.innerHTML = resultsHtml;
                    
                    // Render Loss Chart
                    chartContainer.style.display = "block";
                    const labels = data.losses.map((_, i) => `Epoch ${i+1}`);
                    drawLossChart(labels, data.losses);
                    fetchPipelineInfo();
                } else {
                    resultsPanel.innerHTML = `<p class="info-text text-error">Failed to complete model training.</p>`;
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-graduation-cap"></i> Start Model Training';
                loader.classList.add("hidden");
                resultsPanel.innerHTML = `<p class="info-text text-error">API request error.</p>`;
                console.error(err);
            });
        });

        // --- TAB 5: INFERENCE & YIELD ---
        document.getElementById("btn-run-inference").addEventListener("click", () => {
            const btn = document.getElementById("btn-run-inference");
            const loader = document.getElementById("inference-loader");
            const imgPreview = document.getElementById("inference-img");
            const captionEl = document.getElementById("inference-caption");
            const metricsContainer = document.getElementById("yield-metrics-container");

            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Localizing...';
            loader.classList.remove("hidden");
            imgPreview.src = "";
            captionEl.textContent = "Running neural segmentations...";
            metricsContainer.classList.add("hidden");

            const cropMode = document.getElementById("inference-crop-mode").value;

            fetch(`${API_ROOT}/api/inference`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ crop_mode: cropMode })
            })
            .then(res => res.json())
            .then(data => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-calculator"></i> Run Crop Analysis & Localization';
                loader.classList.add("hidden");

                if (data.status === "success") {
                    imgPreview.src = `data:image/png;base64,${data.image}`;
                    captionEl.textContent = "Localized boundaries of identified fields.";
                    metricsContainer.classList.remove("hidden");

                    // Update summary metrics
                    document.getElementById("inf-fields-count").textContent = data.summary.fields_count;
                    document.getElementById("inf-total-area").textContent = `${data.summary.total_area.toFixed(2)} ha`;
                    document.getElementById("inf-total-yield").textContent = `${data.summary.total_yield.toFixed(1)} t`;
                    document.getElementById("inf-mean-ndvi").textContent = data.summary.mean_ndvi.toFixed(3);

                    // Update detailed table
                    const tbody = document.querySelector("#fields-table tbody");
                    tbody.innerHTML = "";
                    data.table.forEach(row => {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `
                            <td>Field #${row.field_id}</td>
                            <td>${row.area} ha</td>
                            <td>${row.ndvi}</td>
                            <td><span class="status-${row.health.toLowerCase()}">${row.health}</span></td>
                            <td>${row.crop_type}</td>
                            <td><strong>${row.yield} tons</strong></td>
                        `;
                        tbody.appendChild(tr);
                    });
                    
                    fetchPipelineInfo();
                } else {
                    captionEl.textContent = data.message || "Failed to run inference.";
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-calculator"></i> Run Crop Analysis & Localization';
                loader.classList.add("hidden");
                captionEl.textContent = "Request failed.";
                console.error(err);
            });
        });
    }

    // ==========================================
    // BACKEND SERVICE CALLBACKS
    // ==========================================
    function loadSampleDataset(type) {
        const resultsContainer = document.getElementById("catalog-results-container");
        resultsContainer.innerHTML = `<p class="info-text">Acquiring sample ${type} satellite image...</p>`;

        fetch(`${API_ROOT}/api/acquire`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: "load_sample",
                sample_type: type
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                resultsContainer.innerHTML = `<p class="info-text status-healthy"><i class="fa-solid fa-check"></i> ${data.message}. Scene path: <code>${data.current_scene_path}</code></p>`;
                fetchPipelineInfo();
            } else {
                resultsContainer.innerHTML = `<p class="info-text status-stressed">Error loading data.</p>`;
            }
        })
        .catch(err => {
            resultsContainer.innerHTML = `<p class="info-text status-stressed">Request failed.</p>`;
            console.error(err);
        });
    }

    function loadBhoonidhiScene(productId, sensor, btnElement) {
        const originalText = btnElement.textContent;
        btnElement.disabled = true;
        btnElement.textContent = "Loading...";

        fetch(`${API_ROOT}/api/acquire`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: "download",
                product_id: productId,
                sensor: sensor
            })
        })
        .then(res => res.json())
        .then(data => {
            btnElement.disabled = false;
            btnElement.textContent = originalText;

            const resultsContainer = document.getElementById("catalog-results-container");
            if (data.status === "success") {
                resultsContainer.innerHTML = `<p class="info-text status-healthy"><i class="fa-solid fa-check"></i> ${data.message}. Scene path: <code>${data.current_scene_path}</code></p>`;
                fetchPipelineInfo();
            } else {
                resultsContainer.innerHTML = `<p class="info-text status-stressed">Error downloading scene.</p>`;
            }
        })
        .catch(err => {
            btnElement.disabled = false;
            btnElement.textContent = originalText;
            console.error(err);
        });
    }

    function drawLossChart(labels, data) {
        const ctx = document.getElementById("loss-chart").getContext("2d");
        
        if (trainingChartInstance) {
            trainingChartInstance.destroy();
        }

        trainingChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Training Loss',
                    data: data,
                    backgroundColor: 'rgba(0, 153, 255, 0.15)',
                    borderColor: '#0099ff',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: 'var(--text-color)' }
                    }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'var(--text-muted)' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'var(--text-muted)' }
                    }
                }
            }
        });
    }
});
