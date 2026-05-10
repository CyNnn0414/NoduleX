export type RiskLevel = "low" | "intermediate" | "high";
export type NoduleClass =
  | "solid"
  | "part-solid"
  | "ground-glass"
  | "calcified"
  | "benign-pattern"
  | "suspicious";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface NoduleMeasurement {
  longest_diameter_mm: number;
  shortest_diameter_mm: number;
  estimated_volume_mm3: number;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface NoduleFinding {
  id: string;
  slice_index: number;
  bbox: BoundingBox;
  confidence: number;
  classification: NoduleClass;
  malignancy_risk: RiskLevel;
  measurement: NoduleMeasurement;
  reasoning: string;
  accepted: boolean | null;
  clinician_note?: string | null;
  patient_summary: string;
}

export interface PatientProfile {
  patient_id: string;
  name: string;
  age: number;
  sex: string;
  smoking_history: string;
}

export interface StudyRecord {
  id: string;
  patient: PatientProfile;
  metadata: {
    accession_number: string;
    modality: string;
    study_description: string;
    collected_at: string;
    voxel_spacing_mm: [number, number, number];
    slice_count: number;
  };
  status: "uploaded" | "processing" | "ready" | "error";
  findings: NoduleFinding[];
  patient_chat_history?: ChatTurn[];
  basic_diagnosis?: string | null;
  created_at?: string;
  updated_at?: string;
  source_path?: string | null;
}

export interface StudySummary {
  study_id: string;
  patient_name: string;
  nodule_count: number;
  classifications: Record<string, number>;
  highest_risk: RiskLevel;
  basic_diagnosis?: string | null;
}
