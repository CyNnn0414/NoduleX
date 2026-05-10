const LOCAL_SERVER_BASE = "http://127.0.0.1:8000";
const API_BASE = window.location.protocol === "file:" ? `${LOCAL_SERVER_BASE}/api` : "/api";
const BACKEND_POLL_INTERVAL_MS = 5000;

const state = {
  viewMode: "doctor",
  doctorReviewFilter: "pending",
  studies: [],
  selectedStudyId: null,
  selectedFindingId: null,
  currentSliceIndex: 0,
  overlayVisible: true,
  summary: null,
  busy: false,
  offlineMode: false,
  backendConnected: false,
  chatTurns: [],
};

const demoStudy = {
  id: "study_demo_nodx",
  patient: {
    patient_id: "patient_demo_jane_doe",
    name: "Jane Doe",
    age: 63,
    sex: "female",
    smoking_history: "Former smoker, quit 8 years ago",
  },
  metadata: {
    accession_number: "ACC-DEMO-0001",
    modality: "CT",
    study_description: "Chest CT - Demo",
    collected_at: new Date().toISOString(),
    voxel_spacing_mm: [1, 1, 1],
    uploaded_file_count: 248,
    source_slice_count: 248,
    analysis_slice_count: 248,
    slice_count: 248,
    preview_image_url: null,
    slice_image_urls: [],
    analysis_method: "demo",
  },
  status: "ready",
  basic_diagnosis:
    "Demo screening identified two lung nodule candidates. This built-in preview is educational only and does not replace a clinician review.",
  findings: [
    {
      id: "finding_demo_01",
      slice_index: 126,
      bbox: { x: 0.42, y: 0.31, width: 0.08, height: 0.08 },
      confidence: 0.94,
      classification: "solid",
      malignancy_risk: "intermediate",
      measurement: { longest_diameter_mm: 7.8, shortest_diameter_mm: 5.9, estimated_volume_mm3: 176.0 },
      reasoning: "Solid nodule candidate with strong slice-level confidence and consistent contour across adjacent slices.",
      accepted: true,
      patient_summary: "A small solid lung nodule was identified. Your care team can explain what this means in the context of your full scan.",
      preview_image_url: null,
    },
    {
      id: "finding_demo_02",
      slice_index: 133,
      bbox: { x: 0.58, y: 0.47, width: 0.1, height: 0.1 },
      confidence: 0.88,
      classification: "ground-glass",
      malignancy_risk: "low",
      measurement: { longest_diameter_mm: 4.2, shortest_diameter_mm: 3.7, estimated_volume_mm3: 41.0 },
      reasoning: "Ground-glass pattern with lower risk features and smaller measured diameter.",
      accepted: true,
      patient_summary: "A very small faint nodule was seen. These findings can have many causes and need clinician interpretation.",
      preview_image_url: null,
    },
  ],
};

const elements = {
  doctorModeBtn: document.getElementById("doctor-mode-btn"),
  patientModeBtn: document.getElementById("patient-mode-btn"),
  uploadForm: document.getElementById("upload-form"),
  uploadButton: document.getElementById("upload-button"),
  viewerTitle: document.getElementById("viewer-title"),
  overlayButton: document.getElementById("overlay-button"),
  analyzeButton: document.getElementById("analyze-button"),
  viewerEmpty: document.getElementById("viewer-empty"),
  viewerContent: document.getElementById("viewer-content"),
  statusRibbon: document.getElementById("status-ribbon"),
  studySummaryRow: document.getElementById("study-summary-row"),
  sliceControls: document.getElementById("slice-controls"),
  slicePositionLabel: document.getElementById("slice-position-label"),
  sliceFindingsLabel: document.getElementById("slice-findings-label"),
  sliceSlider: document.getElementById("slice-slider"),
  slicePrevButton: document.getElementById("slice-prev-button"),
  sliceNextButton: document.getElementById("slice-next-button"),
  scanViewport: document.getElementById("scan-viewport"),
  scanImage: document.getElementById("scan-image"),
  scanCaption: document.getElementById("scan-caption"),
  findingsTitle: document.getElementById("findings-title"),
  findingsPanel: document.getElementById("findings-panel"),
  riskPill: document.getElementById("risk-pill"),
  findingsContent: document.getElementById("findings-content"),
  chatColumn: document.getElementById("chat-column"),
  chatThread: document.getElementById("chat-thread"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  errorBanner: document.getElementById("error-banner"),
  runtimeBanner: document.getElementById("local-runtime-banner"),
  engineStatusCard: document.getElementById("engine-status-card"),
  engineStatusText: document.getElementById("engine-status-text"),
  reconnectButton: document.getElementById("reconnect-button"),
  fileInput: document.querySelector('input[name="files"]'),
  fileSelectionSummary: document.getElementById("file-selection-summary"),
};

let backendPollHandle = null;

function assetUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//.test(path) || path.startsWith("data:") || path.startsWith("file:")) {
    return path;
  }
  return window.location.protocol === "file:" ? `${LOCAL_SERVER_BASE}${path}` : path;
}

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function api(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  if (!response.ok) {
    let message = "";
    try {
      const data = await response.json();
      message = typeof data.detail === "string" ? data.detail : JSON.stringify(data);
    } catch {
      message = await response.text();
    }
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json();
}

function setRuntimeBanner(message) {
  if (!message) {
    elements.runtimeBanner.classList.add("hidden");
    return;
  }
  elements.runtimeBanner.textContent = message;
  elements.runtimeBanner.classList.remove("hidden");
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.classList.remove("hidden");
}

function clearError() {
  elements.errorBanner.classList.add("hidden");
}

function setConnectionState(isConnected) {
  state.backendConnected = isConnected;
  state.offlineMode = !isConnected;
  elements.engineStatusCard.classList.remove("online", "offline");
  elements.engineStatusCard.classList.add(isConnected ? "online" : "offline");
  elements.engineStatusText.textContent = isConnected
    ? "Connected. You can upload one folder of DICOM CT slices, save it as a study, and run local nodule detection and analysis on this computer."
    : "NoduleX is not connected to its local analysis service yet. Start the local app, then this page will reconnect automatically.";
  elements.reconnectButton.textContent = isConnected ? "Check connection" : "Reconnect";
  if (window.location.protocol === "file:") {
    setRuntimeBanner(
      isConnected
        ? "Connected to the local NoduleX backend at http://127.0.0.1:8000 while viewing this page from disk."
        : "This page is open from disk. Real DICOM upload and nodule detection turn on automatically when the local NoduleX backend at http://127.0.0.1:8000 is available.",
    );
  } else {
    setRuntimeBanner("");
  }
}

function updateActionAvailability() {
  const study = getSelectedStudy();
  const sliceCount = getSliceCount(study);
  const uploadLabel = state.offlineMode ? "Start local app to upload" : "Upload folder and analyze";
  const analyzeLabel = state.offlineMode ? "Start local app to analyze" : "Run AI analysis";
  elements.uploadButton.disabled = state.busy || state.offlineMode;
  elements.analyzeButton.disabled = state.busy || state.offlineMode || !state.selectedStudyId;
  elements.reconnectButton.disabled = state.busy;
  elements.sliceSlider.disabled = state.busy || sliceCount <= 1;
  elements.slicePrevButton.disabled = state.busy || sliceCount <= 1 || state.currentSliceIndex <= 0;
  elements.sliceNextButton.disabled = state.busy || sliceCount <= 1 || state.currentSliceIndex >= sliceCount - 1;
  if (!state.busy) {
    elements.uploadButton.textContent = uploadLabel;
    elements.analyzeButton.textContent = analyzeLabel;
  }
}

function updateFileSelectionSummary() {
  const files = Array.from(elements.fileInput?.files ?? []).filter((file) => file && file.name);
  if (!files.length) {
    elements.fileSelectionSummary.classList.add("hidden");
    elements.fileSelectionSummary.textContent = "";
    return;
  }

  const message =
    files.length === 1
      ? "1 DICOM slice selected. A full CT study usually contains many slices, so detection works best when you choose the whole folder."
      : `${files.length} DICOM slices selected. NoduleX will keep them together as one 3D CT study for detection and analysis.`;
  elements.fileSelectionSummary.textContent = state.offlineMode
    ? `${message} The files are selected, but the local analysis service still needs to be running before upload can start.`
    : message;
  elements.fileSelectionSummary.classList.remove("hidden");
}

function sortSelectedFiles(files) {
  return [...files].sort((left, right) => {
    const leftKey = (left.webkitRelativePath || left.name).toLowerCase();
    const rightKey = (right.webkitRelativePath || right.name).toLowerCase();
    return leftKey.localeCompare(rightKey, undefined, { numeric: true, sensitivity: "base" });
  });
}

async function pingBackend() {
  try {
    const response = await fetch(apiUrl("/health"));
    return response.ok;
  } catch {
    return false;
  }
}

async function refreshBackendConnection({ reloadOnReconnect = false } = {}) {
  const isConnected = await pingBackend();
  const wasOffline = state.offlineMode;
  setConnectionState(isConnected);
  updateActionAvailability();
  updateFileSelectionSummary();
  if (isConnected && reloadOnReconnect && wasOffline) {
    clearError();
    await loadStudies();
  }
  return isConnected;
}

function ensureBackendPolling() {
  if (backendPollHandle || window.location.protocol !== "file:") return;
  backendPollHandle = window.setInterval(() => {
    void refreshBackendConnection({ reloadOnReconnect: true });
  }, BACKEND_POLL_INTERVAL_MS);
}

function fallbackSummary(study) {
  const approved = study.findings.filter((finding) => finding.accepted !== false);
  const classifications = {};
  for (const finding of approved) {
    classifications[finding.classification] = (classifications[finding.classification] ?? 0) + 1;
  }
  const highestRisk = approved.some((finding) => finding.malignancy_risk === "high")
    ? "high"
    : approved.some((finding) => finding.malignancy_risk === "intermediate")
      ? "intermediate"
      : "low";
  return {
    study_id: study.id,
    patient_name: study.patient.name,
    nodule_count: approved.length,
    classifications,
    highest_risk: highestRisk,
    basic_diagnosis: study.basic_diagnosis,
  };
}

function getStudyChatTurns(study) {
  return study?.patient_chat_history?.filter((turn) => turn.content?.trim()) ?? [];
}

function getSelectedStudy() {
  return state.studies.find((study) => study.id === state.selectedStudyId) ?? null;
}

function getSelectedFinding() {
  const study = getSelectedStudy();
  return study?.findings.find((finding) => finding.id === state.selectedFindingId) ?? study?.findings[0] ?? null;
}

function getApprovedFindings() {
  const study = getSelectedStudy();
  return study ? study.findings.filter((finding) => finding.accepted !== false) : [];
}

function getDoctorAcceptedFindings() {
  const study = getSelectedStudy();
  return study ? study.findings.filter((finding) => finding.accepted === true) : [];
}

function getPendingFindings(study) {
  return study ? study.findings.filter((finding) => finding.accepted == null) : [];
}

function getAcceptedFindings(study) {
  return study ? study.findings.filter((finding) => finding.accepted === true) : [];
}

function getRejectedFindings(study) {
  return study ? study.findings.filter((finding) => finding.accepted === false) : [];
}

function getDoctorReviewCollection(study) {
  if (!study) return [];
  if (state.doctorReviewFilter === "accepted") return getAcceptedFindings(study);
  if (state.doctorReviewFilter === "rejected") return getRejectedFindings(study);
  return getPendingFindings(study);
}

function nextReviewFindingForCurrentFilter(study) {
  return getDoctorReviewCollection(study)[0] ?? null;
}

function getSliceUrls(study) {
  return study?.metadata?.slice_image_urls ?? [];
}

function getSliceCount(study) {
  const urls = getSliceUrls(study);
  return urls.length || study?.metadata?.slice_count || 0;
}

function clampSliceIndex(study, sliceIndex) {
  const sliceCount = getSliceCount(study);
  if (!sliceCount) return 0;
  return Math.max(0, Math.min(sliceCount - 1, sliceIndex));
}

function setCurrentSliceIndex(sliceIndex, study = getSelectedStudy()) {
  state.currentSliceIndex = clampSliceIndex(study, sliceIndex);
}

function sortStudiesByRecency(studies) {
  return [...studies].sort((left, right) => {
    const leftTime = Date.parse(left.updated_at || left.created_at || 0) || 0;
    const rightTime = Date.parse(right.updated_at || right.created_at || 0) || 0;
    return rightTime - leftTime;
  });
}

function latestStudy(studies) {
  return sortStudiesByRecency(studies)[0] ?? null;
}

function jumpToBestSlice(study, preferredSliceIndex = null) {
  if (!study) {
    state.currentSliceIndex = 0;
    return;
  }
  const targetSlice =
    preferredSliceIndex ??
    study.findings.find((finding) => finding.id === state.selectedFindingId)?.slice_index ??
    Math.floor(getSliceCount(study) / 2);
  setCurrentSliceIndex(targetSlice, study);
}

function findingsForCurrentSlice(study) {
  if (!study) return [];
  return study.findings.filter((finding) => finding.slice_index === state.currentSliceIndex);
}

function doctorAcceptedFindingsForCurrentSlice(study) {
  if (!study) return [];
  return study.findings.filter((finding) => finding.accepted === true && finding.slice_index === state.currentSliceIndex);
}

function focusFinding(finding) {
  if (!finding) return;
  state.selectedFindingId = finding.id;
  setCurrentSliceIndex(finding.slice_index);
}

function cycleDoctorAcceptedFinding() {
  const accepted = getDoctorAcceptedFindings();
  if (!accepted.length) return;
  const currentIndex = accepted.findIndex((finding) => finding.id === state.selectedFindingId);
  const nextFinding = accepted[(currentIndex + 1 + accepted.length) % accepted.length] ?? accepted[0];
  focusFinding(nextFinding);
}

function setBusy(isBusy, label = "Working...") {
  state.busy = isBusy;
  elements.overlayButton.disabled = false;
  if (isBusy) {
    elements.uploadButton.textContent = label;
    elements.analyzeButton.textContent = label;
  }
  updateActionAvailability();
}

function renderScanImage(study) {
  const sliceUrls = getSliceUrls(study);
  const previewUrl = assetUrl(sliceUrls[state.currentSliceIndex] || study?.metadata?.preview_image_url);
  if (!previewUrl) {
    elements.scanImage.classList.add("hidden");
    elements.scanImage.removeAttribute("src");
    elements.scanViewport.style.removeProperty("aspect-ratio");
    elements.scanViewport.classList.remove("has-image");
    return;
  }
  elements.scanImage.onload = () => {
    const { naturalWidth, naturalHeight } = elements.scanImage;
    if (naturalWidth > 0 && naturalHeight > 0) {
      elements.scanViewport.style.aspectRatio = `${naturalWidth} / ${naturalHeight}`;
    }
    elements.scanViewport.classList.add("has-image");
  };
  elements.scanImage.src = previewUrl;
  elements.scanImage.classList.remove("hidden");
}

function renderViewer() {
  const study = getSelectedStudy();
  const doctorAccepted = getDoctorAcceptedFindings();
  if (state.viewMode === "patient" && doctorAccepted.length && !doctorAccepted.some((item) => item.id === state.selectedFindingId)) {
    state.selectedFindingId = doctorAccepted[0].id;
    state.currentSliceIndex = doctorAccepted[0].slice_index;
  }
  const finding = getSelectedFinding();
  const approved = getApprovedFindings();
  const pendingFindings = getPendingFindings(study);
  const acceptedFindings = getAcceptedFindings(study);
  const rejectedFindings = getRejectedFindings(study);
  const doctorReviewItems = getDoctorReviewCollection(study);
  state.currentSliceIndex = clampSliceIndex(study, state.currentSliceIndex);
  const sliceCount = getSliceCount(study);
  const allFindingsOnCurrentSlice = findingsForCurrentSlice(study);
  const patientVisibleFindingsOnCurrentSlice = doctorAcceptedFindingsForCurrentSlice(study);
  const findingsOnCurrentSlice = state.viewMode === "doctor" ? allFindingsOnCurrentSlice : patientVisibleFindingsOnCurrentSlice;

  elements.viewerTitle.textContent = state.viewMode === "doctor" ? "Doctor Console" : "Patient Scan View";
  elements.findingsTitle.textContent =
    state.viewMode === "doctor"
      ? state.doctorReviewFilter === "accepted"
        ? "Accepted Findings"
        : state.doctorReviewFilter === "rejected"
          ? "Rejected Findings"
          : "Findings Review"
      : "Patient Summary";
  elements.findingsPanel.classList.remove("hidden");
  elements.chatColumn.classList.toggle("hidden", state.viewMode !== "patient");
  elements.analyzeButton.classList.toggle("hidden", state.viewMode !== "doctor");

  if (!study) {
    elements.viewerEmpty.classList.remove("hidden");
    elements.viewerContent.classList.add("hidden");
    elements.sliceControls.classList.add("hidden");
    elements.findingsContent.className = "empty-state";
    elements.findingsContent.textContent = "Upload a study to begin.";
    elements.riskPill.className = "risk-pill hidden";
    updateActionAvailability();
    return;
  }

  elements.viewerEmpty.classList.add("hidden");
  elements.viewerContent.classList.remove("hidden");

  const methodLabel = study.metadata.analysis_method ? study.metadata.analysis_method.toUpperCase() : "LOCAL";
  const uploadedFileCount = study.metadata.uploaded_file_count || study.metadata.source_slice_count || study.metadata.slice_count || 0;
  const sourceSliceCount = study.metadata.source_slice_count || study.metadata.slice_count || 0;
  const analysisSliceCount = study.metadata.analysis_slice_count || study.metadata.slice_count || 0;
  const viewerSliceCount = sliceCount;

  if (state.viewMode === "doctor") {
    elements.statusRibbon.innerHTML = `
      <button type="button" class="status-chip status-chip-button ${state.doctorReviewFilter === "pending" ? "active" : ""}" data-review-filter="pending">
        <span>Needs review</span><strong>${pendingFindings.length}</strong>
      </button>
      <button type="button" class="status-chip status-chip-button ${state.doctorReviewFilter === "accepted" ? "active" : ""}" data-review-filter="accepted">
        <span>Accepted by doctor</span><strong>${acceptedFindings.length}</strong>
      </button>
      <button type="button" class="status-chip status-chip-button ${state.doctorReviewFilter === "rejected" ? "active" : ""}" data-review-filter="rejected">
        <span>Rejected</span><strong>${rejectedFindings.length}</strong>
      </button>
      <div class="status-chip"><span>Analysis</span><strong>${methodLabel}</strong></div>
    `;
    elements.statusRibbon.querySelectorAll("[data-review-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        state.doctorReviewFilter = button.dataset.reviewFilter;
        if (state.doctorReviewFilter === "pending") {
          state.selectedFindingId = pendingFindings[0]?.id ?? state.selectedFindingId;
        } else if (state.doctorReviewFilter === "accepted") {
          state.selectedFindingId = acceptedFindings[0]?.id ?? state.selectedFindingId;
        } else if (state.doctorReviewFilter === "rejected") {
          state.selectedFindingId = rejectedFindings[0]?.id ?? state.selectedFindingId;
        }
        const nextStudy = getSelectedStudy();
        const nextFinding = nextStudy?.findings.find((item) => item.id === state.selectedFindingId);
        if (nextFinding) setCurrentSliceIndex(nextFinding.slice_index);
        renderViewer();
      });
    });
  } else {
    elements.statusRibbon.innerHTML = `
      <button type="button" class="status-chip status-chip-button active" data-patient-accepted="true">
        <span>Doctor-accepted</span><strong>${doctorAccepted.length}</strong>
      </button>
      <div class="status-chip"><span>Analysis</span><strong>${methodLabel}</strong></div>
    `;
    const acceptedButton = elements.statusRibbon.querySelector("[data-patient-accepted]");
    acceptedButton?.addEventListener("click", () => {
      cycleDoctorAcceptedFinding();
      renderViewer();
    });
  }

  const summaryNoduleCount = state.viewMode === "doctor" ? study.findings.length : doctorAccepted.length;
  elements.studySummaryRow.innerHTML = `
    <div><p class="label">Patient</p><strong>${study.patient.name}</strong></div>
    <div><p class="label">Accession</p><strong>${study.metadata.accession_number}</strong></div>
    <div><p class="label">Uploaded DICOM files</p><strong>${uploadedFileCount}</strong></div>
    <div><p class="label">Original CT slices</p><strong>${sourceSliceCount}</strong></div>
    <div><p class="label">Viewer CT slices</p><strong>${viewerSliceCount}</strong></div>
    <div><p class="label">Analysis slices</p><strong>${analysisSliceCount}</strong></div>
    <div><p class="label">${state.viewMode === "doctor" ? "Detected nodules" : "Patient-visible nodules"}</p><strong>${summaryNoduleCount}</strong></div>
  `;

  elements.sliceControls.classList.toggle("hidden", sliceCount <= 0);
  elements.sliceSlider.max = String(Math.max(sliceCount - 1, 0));
  elements.sliceSlider.value = String(clampSliceIndex(study, state.currentSliceIndex));
  elements.slicePositionLabel.textContent = sliceCount
    ? `Viewer slice ${state.currentSliceIndex + 1} of ${sliceCount}`
    : "Slice 0 of 0";
  elements.sliceFindingsLabel.textContent =
    findingsOnCurrentSlice.length === 0
      ? "No detections on this slice"
      : findingsOnCurrentSlice.length === 1
        ? "1 finding on this slice"
        : `${findingsOnCurrentSlice.length} findings on this slice`;

  renderScanImage(study);
  elements.scanViewport.querySelectorAll(".overlay-box").forEach((node) => node.remove());
  if (state.overlayVisible) {
    findingsOnCurrentSlice.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `overlay-box risk-${item.malignancy_risk} ${finding?.id === item.id ? "focused" : ""}`;
      button.style.left = `${item.bbox.x * 100}%`;
      button.style.top = `${item.bbox.y * 100}%`;
      button.style.width = `${item.bbox.width * 100}%`;
      button.style.height = `${item.bbox.height * 100}%`;
      button.innerHTML = `<span>${item.classification}</span>`;
      button.addEventListener("click", () => {
        state.selectedFindingId = item.id;
        setCurrentSliceIndex(item.slice_index);
        renderViewer();
      });
      elements.scanViewport.appendChild(button);
    });
  }

  const captionFinding =
    finding &&
    finding.slice_index === state.currentSliceIndex &&
    (state.viewMode === "doctor" || finding.accepted === true)
      ? finding
      : findingsOnCurrentSlice[0] ?? null;
  elements.scanCaption.innerHTML = captionFinding
    ? `
      <strong>Selected slice ${captionFinding.slice_index}</strong>
      <div class="caption-meta">
        <span>Original CT viewer</span>
        <span>${captionFinding.classification}</span>
        <span>${captionFinding.measurement.longest_diameter_mm.toFixed(1)} mm</span>
        <span>${Math.round(captionFinding.confidence * 100)}% confidence</span>
      </div>
      <p>${captionFinding.reasoning}</p>
    `
    : `
      <strong>Slice ${state.currentSliceIndex + 1}</strong>
      <p>${study.basic_diagnosis || "No AI finding is marked on this slice yet."}</p>
    `;

  if (state.summary) {
    elements.riskPill.textContent = `${state.summary.highest_risk} risk profile`;
    elements.riskPill.className = `risk-pill ${state.summary.highest_risk}`;
  } else {
    elements.riskPill.className = "risk-pill hidden";
  }

  const diagnosisCard = `
    <div class="diagnosis-card">
      <strong>Basic AI screening impression</strong>
      <p>${study.basic_diagnosis || "No screening summary is available yet."}</p>
    </div>
  `;

  if (state.viewMode === "doctor") {
    elements.findingsContent.className = "finding-list";
    elements.findingsContent.innerHTML = diagnosisCard;
    if (!doctorReviewItems.length) {
      const empty = document.createElement("article");
      empty.className = "patient-card";
      empty.innerHTML =
        state.doctorReviewFilter === "pending"
          ? "<h3>Review queue complete</h3><p>All findings in this study have a decision. Use the Accepted by doctor or Rejected buttons above to review those summaries.</p>"
          : state.doctorReviewFilter === "accepted"
            ? "<h3>No accepted findings yet</h3><p>Accepted findings will appear here after you confirm them.</p>"
            : "<h3>No rejected findings yet</h3><p>Rejected findings will appear here after you mark them.</p>";
      elements.findingsContent.appendChild(empty);
    }

    doctorReviewItems.forEach((item) => {
      const card = document.createElement("article");
      card.className = `finding-card ${finding?.id === item.id ? "selected" : ""}`;
      const decisionBadge =
        item.accepted === true
          ? `<div class="review-badge accepted">Accepted</div>`
          : item.accepted === false
            ? `<div class="review-badge rejected">Rejected</div>`
            : "";
      card.innerHTML = `
        <button type="button" class="finding-select">
          <div>
            <strong>${item.classification}</strong>
            <p>Slice ${item.slice_index}</p>
          </div>
          <div class="finding-meta">
            ${decisionBadge}
            <div class="confidence-badge">${Math.round(item.confidence * 100)}%</div>
          </div>
        </button>
        <p>${item.reasoning}</p>
        <div class="measurement-row">
          <span>${item.measurement.longest_diameter_mm.toFixed(1)} mm</span>
          <span>${item.measurement.estimated_volume_mm3.toFixed(0)} mm3</span>
          <span>${item.malignancy_risk} risk</span>
        </div>
        ${
          state.doctorReviewFilter === "pending"
            ? `
              <div class="decision-row">
                <button type="button" data-decision="accept">Accept</button>
                <button type="button" class="ghost" data-decision="reject">Reject</button>
              </div>
            `
            : `
              <div class="decision-summary-row">
                <span>${item.accepted ? "Accepted diagnosis" : "Rejected diagnosis"}</span>
                <div class="decision-row">
                  ${
                    item.accepted
                      ? `<button type="button" class="ghost" data-decision="reject">Change to reject</button>`
                      : `<button type="button" data-decision="accept">Change to accept</button>`
                  }
                </div>
              </div>
            `
        }
      `;
      card.querySelector(".finding-select").addEventListener("click", () => {
        state.selectedFindingId = item.id;
        setCurrentSliceIndex(item.slice_index);
        renderViewer();
      });
      if (card.querySelector('[data-decision="accept"]')) {
        card.querySelector('[data-decision="accept"]').addEventListener("click", () => reviewFinding(item.id, true));
      }
      if (card.querySelector('[data-decision="reject"]')) {
        card.querySelector('[data-decision="reject"]').addEventListener("click", () => reviewFinding(item.id, false));
      }
      elements.findingsContent.appendChild(card);
    });
  } else {
    elements.findingsContent.className = "patient-panel";
    elements.findingsContent.innerHTML = `
      ${
        doctorAccepted.length
          ? `
            <div class="patient-accepted-picker">
              <strong>Choose a doctor-accepted nodule</strong>
              <div class="patient-inline-selector-list">
                ${doctorAccepted
                  .map(
                    (item, index) => `
                      <button
                        type="button"
                        class="patient-inline-selector-button ${state.selectedFindingId === item.id ? "selected" : ""}"
                        data-patient-accepted-id="${item.id}"
                      >
                        Nodule ${index + 1}
                      </button>
                    `,
                  )
                  .join("")}
              </div>
            </div>
          `
          : ""
      }
      <div class="patient-card">
        <h3>What this scan shows</h3>
        <p>${
          state.summary
            ? doctorAccepted.length
              ? `${doctorAccepted.length} doctor-accepted nodules are currently visible in your study. You can ask NoduleX to explain what these findings mean in plain language.`
              : "Your doctor has not accepted any detections for patient display yet."
            : "We are preparing a patient-friendly summary of this study."
        }</p>
      </div>
      <div class="patient-finding-list">
        ${doctorAccepted
          .filter((item) => item.id === state.selectedFindingId)
          .map(
            (item) => `
              <article class="patient-finding selected">
                <strong>${item.classification}</strong>
                <p>${item.patient_summary}</p>
                <small>Largest diameter: ${item.measurement.longest_diameter_mm.toFixed(1)} mm</small>
              </article>
            `,
          )
          .join("")}
      </div>
    `;
    elements.findingsContent.querySelectorAll("[data-patient-accepted-id]").forEach((button) => {
      button.addEventListener("click", () => {
        const selected = doctorAccepted.find((item) => item.id === button.dataset.patientAcceptedId);
        if (!selected) return;
        focusFinding(selected);
        renderViewer();
      });
    });
  }

  updateActionAvailability();
}

function renderChat() {
  elements.chatThread.innerHTML = "";
  if (!state.chatTurns.length) {
    const article = document.createElement("article");
    article.className = "chat-bubble assistant";
    article.textContent =
      "Ask a question about this patient's approved findings, sizes, risk wording, or basic lung nodule terminology.";
    elements.chatThread.appendChild(article);
    return;
  }
  state.chatTurns.forEach((turn) => {
    const article = document.createElement("article");
    article.className = `chat-bubble ${turn.role}`;
    article.textContent = turn.content;
    elements.chatThread.appendChild(article);
  });
}

function syncUploadForm(study) {
  if (!study) return;
  elements.uploadForm.elements.patient_name.value = study.patient.name ?? "";
  elements.uploadForm.elements.smoking_history.value = study.patient.smoking_history ?? "";
}

async function loadStudies() {
  try {
    clearError();
    const data = await api("/studies");
    setConnectionState(true);
    state.studies = sortStudiesByRecency(data.studies);
    const selectedStudy = state.studies.find((study) => study.id === state.selectedStudyId) ?? latestStudy(state.studies);
    state.selectedStudyId = selectedStudy?.id ?? null;
    state.selectedFindingId = selectedStudy?.findings[0]?.id ?? null;
    if (state.selectedStudyId) {
      await loadStudy(state.selectedStudyId);
    } else {
      renderViewer();
    }
    updateActionAvailability();
    updateFileSelectionSummary();
  } catch (error) {
    setConnectionState(false);
    state.studies = [structuredClone(demoStudy)];
    state.selectedStudyId = state.studies[0].id;
    state.selectedFindingId = state.studies[0].findings[0].id;
    state.chatTurns = [];
    jumpToBestSlice(state.studies[0], state.studies[0].findings[0]?.slice_index ?? null);
    state.summary = fallbackSummary(state.studies[0]);
    renderViewer();
    renderChat();
    updateActionAvailability();
    updateFileSelectionSummary();
    showError("Running in demo mode because the local NoduleX service is not connected yet.");
  }
}

async function loadStudy(studyId) {
  if (state.offlineMode) {
    state.summary = fallbackSummary(getSelectedStudy());
    renderViewer();
    return;
  }
  try {
    const data = await api(`/studies/${studyId}`);
    const study = data.study;
    state.studies = [study, ...state.studies.filter((item) => item.id !== study.id)];
    state.selectedStudyId = study.id;
    state.selectedFindingId = state.selectedFindingId ?? study.findings[0]?.id ?? null;
    state.chatTurns = getStudyChatTurns(study);
    jumpToBestSlice(study, study.findings.find((item) => item.id === state.selectedFindingId)?.slice_index ?? null);
    await loadSummary(studyId);
    syncUploadForm(study);
    if (study.status === "error" && study.basic_diagnosis) {
      showError(study.basic_diagnosis);
    } else {
      clearError();
    }
    renderViewer();
    renderChat();
  } catch (error) {
    showError(error.message);
  }
}

async function loadSummary(studyId) {
  if (state.offlineMode) {
    state.summary = fallbackSummary(getSelectedStudy());
    return;
  }
  try {
    state.summary = await api(`/studies/${studyId}/summary`);
  } catch {
    state.summary = null;
  }
}

async function analyzeSelectedStudy() {
  if (!state.selectedStudyId) return;
  if (state.offlineMode) {
    const reconnected = await refreshBackendConnection({ reloadOnReconnect: true });
    if (!reconnected) {
      showError("The local NoduleX service is offline, so analysis cannot start yet. Start the local app and this page will reconnect automatically.");
      return;
    }
  }
  try {
    setBusy(true, "Analyzing...");
    const data = await api(`/studies/${state.selectedStudyId}/analyze`, { method: "POST" });
    const study = data.study;
    state.studies = [study, ...state.studies.filter((item) => item.id !== study.id)];
    state.selectedFindingId = study.findings[0]?.id ?? null;
    jumpToBestSlice(study, study.findings[0]?.slice_index ?? null);
    await loadSummary(study.id);
    syncUploadForm(study);
    renderViewer();
    if (study.status === "error" && study.basic_diagnosis) {
      showError(study.basic_diagnosis);
    } else {
      clearError();
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

async function reviewFinding(findingId, accepted) {
  if (!state.selectedStudyId) return;
  if (state.offlineMode) {
    const study = getSelectedStudy();
    const finding = study?.findings.find((item) => item.id === findingId);
    if (finding) {
      finding.accepted = accepted;
      state.summary = fallbackSummary(study);
      const nextFinding = nextReviewFindingForCurrentFilter(study);
      state.selectedFindingId = nextFinding?.id ?? null;
      if (nextFinding) setCurrentSliceIndex(nextFinding.slice_index);
      renderViewer();
    }
    return;
  }
  try {
    const data = await api(`/studies/${state.selectedStudyId}/findings/${findingId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ accepted, clinician_note: "" }),
    });
    const study = data.study;
    state.studies = [study, ...state.studies.filter((item) => item.id !== study.id)];
    await loadSummary(study.id);
    const nextFinding = nextReviewFindingForCurrentFilter(study);
    state.selectedFindingId = nextFinding?.id ?? null;
    if (nextFinding) setCurrentSliceIndex(nextFinding.slice_index);
    renderViewer();
  } catch (error) {
    showError(error.message);
  }
}

async function uploadStudy(event) {
  event.preventDefault();
  if (state.offlineMode) {
    const reconnected = await refreshBackendConnection({ reloadOnReconnect: true });
    if (!reconnected) {
      showError("The local NoduleX service is offline, so upload cannot start yet. Start the local app and this page will reconnect automatically.");
      return;
    }
  }
  try {
    const files = sortSelectedFiles(Array.from(elements.fileInput.files ?? []).filter((file) => file && file.name));
    if (!files.length) {
      showError("Please choose DICOM files to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("patient_name", elements.uploadForm.elements.patient_name.value);
    formData.append("patient_age", "0");
    formData.append("patient_sex", "other");
    formData.append("smoking_history", elements.uploadForm.elements.smoking_history.value);
    files.forEach((file) => {
      formData.append("files", file, file.webkitRelativePath || file.name);
    });

    setBusy(true, "Uploading folder...");
    const response = await fetch(apiUrl("/studies"), { method: "POST", body: formData });
    if (!response.ok) throw new Error((await response.text()) || "Upload failed.");
    const data = await response.json();
    const study = data.study;
    state.studies = [study];
    state.selectedStudyId = study.id;
    state.selectedFindingId = null;
    state.currentSliceIndex = 0;
    clearError();
    await loadStudy(study.id);
    setBusy(true, "Analyzing study...");
    await analyzeSelectedStudy();
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

async function submitChat(event) {
  event.preventDefault();
  const question = elements.chatInput.value.trim();
  if (!state.selectedStudyId || !question) return;
  state.chatTurns.push({ role: "user", content: question });
  elements.chatInput.value = "";
  renderChat();
  if (state.offlineMode) {
    const study = getSelectedStudy();
    const summary = fallbackSummary(study);
    state.chatTurns.push({
      role: "assistant",
      content: `This demo currently shows ${summary.nodule_count} approved nodules. The screening summary says: ${study.basic_diagnosis}\n\nThis demo answer is educational only and does not replace your clinician.`,
    });
    renderChat();
    return;
  }
  try {
    const reply = await api(`/chat/patient/${state.selectedStudyId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const assistantReply = `${reply.answer}\n\n${reply.safety_note}`;
    state.chatTurns.push({ role: "assistant", content: assistantReply });
    const study = getSelectedStudy();
    if (study) {
      study.patient_chat_history = [...(study.patient_chat_history ?? []), { role: "user", content: question }, { role: "assistant", content: assistantReply }];
    }
  } catch (error) {
    state.chatTurns.push({ role: "assistant", content: error.message });
  }
  renderChat();
}

elements.doctorModeBtn.addEventListener("click", () => {
  state.viewMode = "doctor";
  elements.doctorModeBtn.classList.add("active");
  elements.patientModeBtn.classList.remove("active");
  renderViewer();
});

elements.patientModeBtn.addEventListener("click", () => {
  state.viewMode = "patient";
  elements.patientModeBtn.classList.add("active");
  elements.doctorModeBtn.classList.remove("active");
  renderViewer();
});

elements.overlayButton.addEventListener("click", () => {
  state.overlayVisible = !state.overlayVisible;
  elements.overlayButton.textContent = state.overlayVisible ? "Hide overlay" : "Show overlay";
  renderViewer();
});

elements.analyzeButton.addEventListener("click", () => {
  void analyzeSelectedStudy();
});

elements.sliceSlider.addEventListener("input", (event) => {
  setCurrentSliceIndex(Number(event.target.value));
  renderViewer();
});

elements.slicePrevButton.addEventListener("click", () => {
  setCurrentSliceIndex(state.currentSliceIndex - 1);
  renderViewer();
});

elements.sliceNextButton.addEventListener("click", () => {
  setCurrentSliceIndex(state.currentSliceIndex + 1);
  renderViewer();
});

elements.reconnectButton.addEventListener("click", () => {
  void refreshBackendConnection({ reloadOnReconnect: true });
});

elements.uploadForm.addEventListener("submit", uploadStudy);
elements.fileInput.addEventListener("change", updateFileSelectionSummary);
elements.chatForm.addEventListener("submit", submitChat);

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.chatInput.value = button.dataset.prompt;
  });
});

renderChat();
setConnectionState(false);
updateActionAvailability();
updateFileSelectionSummary();
ensureBackendPolling();
void loadStudies();
