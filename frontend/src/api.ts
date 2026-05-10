import { StudyRecord, StudySummary } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = "";
    try {
      const data = (await response.json()) as { detail?: string };
      message = typeof data.detail === "string" ? data.detail : JSON.stringify(data);
    } catch {
      message = await response.text();
    }
    throw new Error(message || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchStudies(): Promise<StudyRecord[]> {
  const response = await fetch(`${API_BASE}/studies`);
  const data = await parseJson<{ studies: StudyRecord[] }>(response);
  return data.studies;
}

export async function fetchStudy(studyId: string): Promise<StudyRecord> {
  const response = await fetch(`${API_BASE}/studies/${studyId}`);
  const data = await parseJson<{ study: StudyRecord }>(response);
  return data.study;
}

export async function analyzeStudy(studyId: string): Promise<StudyRecord> {
  const response = await fetch(`${API_BASE}/studies/${studyId}/analyze`, {
    method: "POST",
  });
  const data = await parseJson<{ study: StudyRecord }>(response);
  return data.study;
}

export async function reviewFinding(studyId: string, findingId: string, accepted: boolean, clinicianNote?: string): Promise<StudyRecord> {
  const response = await fetch(`${API_BASE}/studies/${studyId}/findings/${findingId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      accepted,
      clinician_note: clinicianNote || null,
    }),
  });
  const data = await parseJson<{ study: StudyRecord }>(response);
  return data.study;
}

export async function fetchSummary(studyId: string): Promise<StudySummary> {
  const response = await fetch(`${API_BASE}/studies/${studyId}/summary`);
  return parseJson<StudySummary>(response);
}

export async function askPatientChat(studyId: string, question: string): Promise<{ answer: string; citations: string[]; safety_note: string }> {
  const response = await fetch(`${API_BASE}/chat/patient/${studyId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return parseJson(response);
}

export async function uploadStudy(formData: FormData): Promise<StudyRecord> {
  const response = await fetch(`${API_BASE}/studies`, {
    method: "POST",
    body: formData,
  });
  const data = await parseJson<{ study: StudyRecord }>(response);
  return data.study;
}
