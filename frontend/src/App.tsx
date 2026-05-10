import { FormEvent, useEffect, useState } from "react";
import { analyzeStudy, askPatientChat, fetchStudies, fetchStudy, fetchSummary, reviewFinding, uploadStudy } from "./api";
import { ChatTurn, NoduleFinding, StudyRecord, StudySummary } from "./types";

type ViewMode = "doctor" | "patient";

const initialUploadState = {
  patientName: "Jane Doe",
  patientAge: "63",
  patientSex: "female",
  smokingHistory: "Former smoker, quit 8 years ago",
};

const patientPrompts = [
  "How many nodules were found in this scan?",
  "What does ground-glass nodule mean?",
  "What size are the approved nodules?",
];

function getStudyChatTurns(study: StudyRecord | null): ChatTurn[] {
  return study?.patient_chat_history?.filter((turn) => turn.content.trim()) ?? [];
}

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>("doctor");
  const [studies, setStudies] = useState<StudyRecord[]>([]);
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null);
  const [summary, setSummary] = useState<StudySummary | null>(null);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("What does a lung nodule mean?");
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [uploadState, setUploadState] = useState(initialUploadState);
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null);

  useEffect(() => {
    void loadStudies();
  }, []);

  useEffect(() => {
    if (!selectedStudyId) {
      return;
    }
    void loadStudy(selectedStudyId);
    void loadSummary(selectedStudyId);
  }, [selectedStudyId]);

  const selectedStudy = studies.find((study) => study.id === selectedStudyId) ?? null;
  const selectedFinding =
    selectedStudy?.findings.find((finding) => finding.id === selectedFindingId) ?? selectedStudy?.findings[0] ?? null;
  const approvedFindings = selectedStudy?.findings.filter((finding) => finding.accepted !== false) ?? [];
  const acceptedCount = selectedStudy?.findings.filter((finding) => finding.accepted === true).length ?? 0;
  const rejectedCount = selectedStudy?.findings.filter((finding) => finding.accepted === false).length ?? 0;

  useEffect(() => {
    setChatTurns(getStudyChatTurns(selectedStudy));
  }, [selectedStudy]);

  async function loadStudies() {
    try {
      const loadedStudies = await fetchStudies();
      setStudies(loadedStudies);
      if (!selectedStudyId && loadedStudies[0]) {
        setSelectedStudyId(loadedStudies[0].id);
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load studies.");
    }
  }

  async function loadStudy(studyId: string) {
    try {
      const study = await fetchStudy(studyId);
      setStudies((current) => {
        const withoutStudy = current.filter((item) => item.id !== study.id);
        return [study, ...withoutStudy];
      });
      if (!selectedFindingId && study.findings[0]) {
        setSelectedFindingId(study.findings[0].id);
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load this study.");
    }
  }

  async function loadSummary(studyId: string) {
    try {
      setSummary(await fetchSummary(studyId));
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Unable to load study summary.");
    }
  }

  async function handleAnalyze() {
    if (!selectedStudyId) {
      return;
    }
    try {
      setBusy(true);
      const updated = await analyzeStudy(selectedStudyId);
      setSelectedFindingId(updated.findings[0]?.id ?? null);
      setStudies((current) => [updated, ...current.filter((item) => item.id !== updated.id)]);
      await loadSummary(updated.id);
    } catch (analyzeError) {
      setError(analyzeError instanceof Error ? analyzeError.message : "Analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleFindingReview(finding: NoduleFinding, accepted: boolean) {
    if (!selectedStudyId) {
      return;
    }
    try {
      const updated = await reviewFinding(selectedStudyId, finding.id, accepted, finding.clinician_note ?? "");
      setStudies((current) => [updated, ...current.filter((item) => item.id !== updated.id)]);
      await loadSummary(updated.id);
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "Unable to update this finding.");
    }
  }

  async function handlePatientChatSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedStudyId || !chatInput.trim()) {
      return;
    }

    const question = chatInput.trim();
    setChatTurns((current) => [...current, { role: "user", content: question }]);
    setChatInput("");

    try {
      const reply = await askPatientChat(selectedStudyId, question);
      const assistantReply = `${reply.answer}\n\n${reply.safety_note}`;
      setStudies((current) =>
        current.map((study) =>
          study.id === selectedStudyId
            ? {
                ...study,
                patient_chat_history: [
                  ...(study.patient_chat_history ?? []),
                  { role: "user", content: question },
                  { role: "assistant", content: assistantReply },
                ],
              }
            : study,
        ),
      );
      setChatTurns((current) => [...current, { role: "assistant", content: assistantReply }]);
    } catch (chatError) {
      setChatTurns((current) => [
        ...current,
        {
          role: "assistant",
          content: chatError instanceof Error ? chatError.message : "The patient assistant is unavailable right now.",
        },
      ]);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!uploadFiles?.length) {
      setError("Please choose DICOM files to upload.");
      return;
    }

    const formData = new FormData();
    Array.from(uploadFiles).forEach((file) => formData.append("files", file));
    formData.append("patient_name", uploadState.patientName);
    formData.append("patient_age", uploadState.patientAge);
    formData.append("patient_sex", uploadState.patientSex);
    formData.append("smoking_history", uploadState.smokingHistory);

    try {
      setBusy(true);
      const study = await uploadStudy(formData);
      setStudies((current) => [study, ...current]);
      setSelectedStudyId(study.id);
      setSelectedFindingId(null);
      setError(null);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-badge">NX</div>
          <div>
            <p className="eyebrow">AI Lung Nodule Intelligence</p>
            <h1>NoduleX</h1>
          </div>
        </div>
        <div className="view-toggle">
          <button className={viewMode === "doctor" ? "active" : ""} onClick={() => setViewMode("doctor")}>
            Doctor Console
          </button>
          <button className={viewMode === "patient" ? "active" : ""} onClick={() => setViewMode("patient")}>
            Patient Companion
          </button>
        </div>
      </header>

      <section className="hero-band">
        <div className="hero-card hero-primary">
          <p className="eyebrow">Clinical Workflow</p>
          <h2>Count, measure, classify, review, and explain chest CT nodules in one workspace.</h2>
          <p>
            NoduleX combines YOLO-based slice analysis, clinician review controls, patient-safe summaries, and a study-grounded chat assistant designed for lung nodule follow-up.
          </p>
          <div className="hero-tags">
            <span>DICOM to YOLO pipeline</span>
            <span>Doctor + patient views</span>
            <span>OpenAI-backed chat</span>
          </div>
        </div>
        <div className="metric-card">
          <span>Sensitivity goal</span>
          <strong>90% for nodules &gt;= 3 mm</strong>
        </div>
        <div className="metric-card">
          <span>False positive goal</span>
          <strong>&lt; 1 per scan</strong>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="workspace">
        <aside className="study-rail">
          <section className="panel">
            <div className="panel-header">
              <h2>Studies</h2>
              <span>{studies.length}</span>
            </div>
            <div className="panel-note">
              Review uploaded studies, open a patient record, and jump directly into the AI overlay workflow.
            </div>
            <div className="study-list">
              {studies.map((study) => (
                <button
                  key={study.id}
                  className={`study-item ${selectedStudyId === study.id ? "selected" : ""}`}
                  onClick={() => {
                    setSelectedStudyId(study.id);
                    setSelectedFindingId(study.findings[0]?.id ?? null);
                  }}
                >
                  <strong>{study.patient.name}</strong>
                  <span>{study.metadata.study_description}</span>
                  <small>{study.status}</small>
                </button>
              ))}
              {!studies.length ? <p className="empty-state">Upload a study or seed the backend demo data to begin.</p> : null}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2>Upload CT Study</h2>
            </div>
            <div className="panel-note">Import a DICOM chest CT series and attach patient context for NoduleX review.</div>
            <form className="upload-form" onSubmit={handleUpload}>
              <label>
                Patient name
                <input value={uploadState.patientName} onChange={(event) => setUploadState((current) => ({ ...current, patientName: event.target.value }))} />
              </label>
              <label>
                Age
                <input value={uploadState.patientAge} onChange={(event) => setUploadState((current) => ({ ...current, patientAge: event.target.value }))} />
              </label>
              <label>
                Sex
                <select value={uploadState.patientSex} onChange={(event) => setUploadState((current) => ({ ...current, patientSex: event.target.value }))}>
                  <option value="female">Female</option>
                  <option value="male">Male</option>
                  <option value="other">Other</option>
                </select>
              </label>
              <label>
                Smoking history
                <input value={uploadState.smokingHistory} onChange={(event) => setUploadState((current) => ({ ...current, smokingHistory: event.target.value }))} />
              </label>
              <label>
                DICOM files
                <input type="file" multiple onChange={(event) => setUploadFiles(event.target.files)} />
              </label>
              <button type="submit" disabled={busy}>Upload study</button>
            </form>
          </section>
        </aside>

        <section className="main-stage">
          <section className="panel viewer-panel">
            <div className="panel-header">
              <h2>{viewMode === "doctor" ? "Doctor Console" : "Patient Scan View"}</h2>
              <div className="viewer-actions">
                <button onClick={() => setOverlayVisible((current) => !current)}>
                  {overlayVisible ? "Hide overlay" : "Show overlay"}
                </button>
                {viewMode === "doctor" ? (
                  <button onClick={handleAnalyze} disabled={busy || !selectedStudyId}>
                    {busy ? "Analyzing..." : "Run AI analysis"}
                  </button>
                ) : null}
              </div>
            </div>

            {selectedStudy ? (
              <>
                <div className="status-ribbon">
                  <div className="status-chip">
                    <span>Approved</span>
                    <strong>{approvedFindings.length}</strong>
                  </div>
                  <div className="status-chip">
                    <span>Accepted by doctor</span>
                    <strong>{acceptedCount}</strong>
                  </div>
                  <div className="status-chip">
                    <span>Rejected</span>
                    <strong>{rejectedCount}</strong>
                  </div>
                  <div className="status-chip">
                    <span>Overlay</span>
                    <strong>{overlayVisible ? "On" : "Off"}</strong>
                  </div>
                </div>

                <div className="study-summary-row">
                  <div>
                    <p className="label">Patient</p>
                    <strong>{selectedStudy.patient.name}</strong>
                  </div>
                  <div>
                    <p className="label">Accession</p>
                    <strong>{selectedStudy.metadata.accession_number}</strong>
                  </div>
                  <div>
                    <p className="label">Slices</p>
                    <strong>{selectedStudy.metadata.slice_count}</strong>
                  </div>
                  <div>
                    <p className="label">{viewMode === "doctor" ? "Detected nodules" : "Patient-visible nodules"}</p>
                    <strong>{viewMode === "doctor" ? selectedStudy.findings.length : approvedFindings.length}</strong>
                  </div>
                </div>

                <div className="scan-stage">
                  <div className="scan-viewport">
                    <div className="scan-glow" />
                    <div className="scan-core" />
                    {overlayVisible
                      ? selectedStudy.findings.map((finding) => (
                          <button
                            key={finding.id}
                            className={`overlay-box risk-${finding.malignancy_risk} ${selectedFinding?.id === finding.id ? "focused" : ""}`}
                            style={{
                              left: `${finding.bbox.x * 100}%`,
                              top: `${finding.bbox.y * 100}%`,
                              width: `${finding.bbox.width * 100}%`,
                              height: `${finding.bbox.height * 100}%`,
                            }}
                            onClick={() => setSelectedFindingId(finding.id)}
                          >
                            <span>{finding.classification}</span>
                          </button>
                        ))
                      : null}
                  </div>

                  <div className="scan-caption">
                    {selectedFinding ? (
                      <>
                        <strong>Selected slice {selectedFinding.slice_index}</strong>
                        <div className="caption-meta">
                          <span>{selectedFinding.classification}</span>
                          <span>{selectedFinding.measurement.longest_diameter_mm.toFixed(1)} mm</span>
                          <span>{Math.round(selectedFinding.confidence * 100)}% confidence</span>
                        </div>
                        <p>{selectedFinding.reasoning}</p>
                      </>
                    ) : (
                      <p>No AI finding selected yet.</p>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-state large">Choose a study to open the viewer.</div>
            )}
          </section>

          <section className="panel findings-panel">
            <div className="panel-header">
              <h2>{viewMode === "doctor" ? "Findings Review" : "Patient Summary"}</h2>
              {summary ? <span className={`risk-pill ${summary.highest_risk}`}>{summary.highest_risk} risk profile</span> : null}
            </div>

            {selectedStudy && viewMode === "doctor" ? (
              <div className="finding-list">
                {selectedStudy.findings.map((finding) => (
                  <article key={finding.id} className={`finding-card ${selectedFinding?.id === finding.id ? "selected" : ""}`}>
                    <button className="finding-select" onClick={() => setSelectedFindingId(finding.id)}>
                      <div>
                        <strong>{finding.classification}</strong>
                        <p>Slice {finding.slice_index}</p>
                      </div>
                      <div className="confidence-badge">{Math.round(finding.confidence * 100)}%</div>
                    </button>
                    <p>{finding.reasoning}</p>
                    <div className="measurement-row">
                      <span>{finding.measurement.longest_diameter_mm.toFixed(1)} mm</span>
                      <span>{finding.measurement.estimated_volume_mm3.toFixed(0)} mm3</span>
                      <span>{finding.malignancy_risk} risk</span>
                    </div>
                    <div className="decision-row">
                      <button onClick={() => void handleFindingReview(finding, true)}>Accept</button>
                      <button className="ghost" onClick={() => void handleFindingReview(finding, false)}>
                        Reject
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            ) : selectedStudy && viewMode === "patient" ? (
              <div className="patient-panel">
                <div className="patient-card">
                  <h3>What this scan shows</h3>
                  <p>
                    {summary
                      ? `${summary.nodule_count} approved nodules are currently visible in your study. You can ask NoduleX to explain what these findings mean in plain language.`
                      : "We are preparing a patient-friendly summary of this study."}
                  </p>
                </div>
                <div className="patient-finding-list">
                  {approvedFindings.map((finding) => (
                      <article key={finding.id} className="patient-finding">
                        <strong>{finding.classification}</strong>
                        <p>{finding.patient_summary}</p>
                        <small>Largest diameter: {finding.measurement.longest_diameter_mm.toFixed(1)} mm</small>
                      </article>
                    ))}
                </div>
              </div>
            ) : (
              <div className="empty-state">No study selected.</div>
            )}
          </section>
        </section>

        {viewMode === "patient" ? (
          <aside className="chat-column">
            <section className="panel chat-panel">
              <div className="panel-header">
                <h2>Patient Chat</h2>
                <span>Study-grounded</span>
              </div>
              <div className="panel-note">
                NoduleX answers using this patient’s approved findings plus basic lung nodule education.
              </div>
              <div className="prompt-row">
                {patientPrompts.map((prompt) => (
                  <button key={prompt} className="prompt-chip" onClick={() => setChatInput(prompt)} type="button">
                    {prompt}
                  </button>
                ))}
              </div>
              <div className="chat-thread">
                {chatTurns.length ? (
                  chatTurns.map((turn, index) => (
                    <article key={`${turn.role}-${index}`} className={`chat-bubble ${turn.role}`}>
                      {turn.content}
                    </article>
                  ))
                ) : (
                  <article className="chat-bubble assistant">
                    Ask a question about this patient’s approved findings, sizes, risk wording, or basic lung nodule terminology.
                  </article>
                )}
              </div>
              <form className="chat-form" onSubmit={handlePatientChatSubmit}>
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="Ask about this patient’s nodules, scan summary, or general lung nodule basics."
                />
                <button type="submit" disabled={!selectedStudyId}>Ask</button>
              </form>
            </section>
          </aside>
        ) : null}
      </main>
    </div>
  );
}

export default App;
