export type RedFlagType = "danger" | "warning";

export interface RedFlag {
  type: RedFlagType;
  title: string;
  message: string;
  suggestion: string;
}

export interface IschemiaResult {
  /** 0-100 — probabilidade estimada pela IA / regras. */
  probability: number;
  detected: boolean;
  diagnosis: string | null;
  wall: string | null;
  leads: string | null;
  artery: string | null;
}

export type ArrhythmiaUrgency = "emergency" | "warning" | "info";

export interface ArrhythmiaResult {
  detected: boolean;
  name: string | null;
  location: string | null;
  characteristics: string[];
  urgency: ArrhythmiaUrgency | null;
  urgencyMessage: string | null;
}

export interface EcgReport {
  id: string;
  createdAt: Date;
  imageUrl: string;
  ischemia: IschemiaResult;
  arrhythmia: ArrhythmiaResult;
  redFlags: RedFlag[];
  reportText: string;
}
