export interface EcgAnalysisSummary {
  id: string;
  imageUrl: string;
  createdAt: Date;
  reportSummary: string | null;
}

export interface DashboardStats {
  totalExams: number;
  thisMonth: number;
  lastExamAt: Date | null;
}

export interface NavItem {
  label: string;
  href: string;
  icon: string;
}
