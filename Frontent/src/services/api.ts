// Frontent/src/services/api.ts

// API configuration and utilities
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

export interface ChatRequest {
  sessionId: string;
  message: string;
}

export interface ChatResponse {
  id: string;
  content: string;
  sessionId: string;
  progress: string[];
  artifactIndex: number;
  execStatus?: string | null;
}

export interface UploadResponse {
  sessionId: string;
  fileId: string;
  status: string;
  rows?: number;
  columns?: string[];
  sample?: any[];
  message?: string;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // Upload files to backend
  async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Upload failed: ${response.statusText} - ${errorText}`);
    }

    return response.json();
  }

  // Send chat message to backend
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.statusText}`);
    }

    return response.json();
  }

  // Stream chat responses (for real-time agent responses)
  streamMessageWS(request: ChatRequest): WebSocket {
    // This logic correctly handles both http -> ws and https -> wss
    const wsUrl = this.baseUrl.replace(/^http/, 'ws');
    const socket = new WebSocket(`${wsUrl}/api/chat/ws`);

    socket.addEventListener('open', () => {
      socket.send(JSON.stringify(request));
    });

    return socket;
  }
}

export const apiService = new ApiService();