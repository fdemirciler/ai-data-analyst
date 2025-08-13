// Extended types for backend integration
export interface BackendFile {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadedAt: string;
  status: 'uploading' | 'uploaded' | 'analyzing' | 'analyzed' | 'error';
  analysisResult?: {
    summary: string;
    insights: string[];
    dataTypes: Record<string, string>;
    rowCount?: number;
    columnCount?: number;
    errors?: string[];
  };
  url?: string; // For frontend preview
}

export interface BackendMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string; // ISO string from backend
  sessionId?: string;
  files?: BackendFile[];
  metadata?: {
    processingTime?: number;
    agentSteps?: string[];
    confidence?: number;
  };
}

export interface ChatSession {
  id: string;
  createdAt: string;
  updatedAt: string;
  messages: BackendMessage[];
  files: BackendFile[];
  status: 'active' | 'completed' | 'error';
}

export interface AnalysisTask {
  id: string;
  fileId: string;
  type: 'summary' | 'insights' | 'visualization' | 'query';
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: string;
  createdAt: string;
  completedAt?: string;
}
