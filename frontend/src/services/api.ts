// API configuration and utilities
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';

interface SessionInfo {
  session_id: string;
  status: string;
  created_at: string;
}

interface UploadResult {
  filename: string;
  rows: number;
  columns: number;
  file_id: string;
  upload_time: string;
  message?: string;
}

interface ExecutionResult {
  success: boolean;
  output: string;
  error?: string;
  execution_time: number;
  memory_used: number;
  warnings?: string[];
  globals_after?: any;
}

interface CodeValidationResult {
  is_safe: boolean;
  violations: string[];
  warnings: string[];
  sanitized_code?: string;
}

export interface ChatRequest {
  message: string;
  files?: File[];
  sessionId?: string;
}

export interface ChatResponse {
  id: string;
  content: string;
  timestamp: string;
  files?: {
    id: string;
    name: string;
    analysisResult?: string;
  }[];
}

export interface UploadResponse {
  fileId: string;
  fileName: string;
  status: 'uploaded' | 'processing' | 'analyzed';
  analysisResult?: string;
}

class ApiService {
  private baseUrl: string;
  private sessionId: string | null = null;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async ensureSession(): Promise<string> {
    if (!this.sessionId) {
      await this.createSession();
    }
    return this.sessionId!;
  }

  async createSession(): Promise<{ success: boolean; data?: SessionInfo; error?: string }> {
    try {
      console.log('ApiService: Creating new session at', `${this.baseUrl}/api/sessions`);
      const response = await fetch(`${this.baseUrl}/api/sessions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      console.log('ApiService: Session creation response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : (errorData.detail && typeof errorData.detail === 'object'
            ? JSON.stringify(errorData.detail)
            : errorData.message || `Session creation failed: ${response.statusText}`);
        throw new Error(errorMessage);
      }

      const data: SessionInfo = await response.json();
      console.log('ApiService: Session created:', data);
      this.sessionId = data.session_id;
      return { success: true, data };
    } catch (error) {
      console.error("ApiService: Session creation error:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Session creation failed"
      };
    }
  }

  async uploadFiles(files: File[]): Promise<{ success: boolean; data?: UploadResult[]; error?: string }> {
    try {
      console.log('ApiService: Starting upload for', files.length, 'files');
      const sessionId = await this.ensureSession();
      console.log('ApiService: Using session ID:', sessionId);
      const results: UploadResult[] = [];

      for (const file of files) {
        console.log('ApiService: Uploading file:', file.name);
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/upload`, {
          method: "POST",
          body: formData,
        });

        console.log('ApiService: Upload response status:', response.status);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const errorMessage = typeof errorData.detail === 'string'
            ? errorData.detail
            : (errorData.detail && typeof errorData.detail === 'object'
              ? JSON.stringify(errorData.detail)
              : errorData.message || `Upload failed for ${file.name}: ${response.statusText}`);
          throw new Error(errorMessage);
        }

        const result: UploadResult = await response.json();
        console.log('ApiService: Upload result:', result);
        results.push(result);
      }

      return { success: true, data: results };
    } catch (error) {
      console.error('ApiService: Upload error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Upload failed"
      };
    }
  }

  async executeCode(code: string): Promise<{ success: boolean; data?: ExecutionResult; error?: string }> {
    try {
      const sessionId = await this.ensureSession();

      const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/execute`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          code: code,
          context_variables: {}
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : (errorData.detail && typeof errorData.detail === 'object'
            ? JSON.stringify(errorData.detail)
            : errorData.message || `Execution failed: ${response.statusText}`);
        throw new Error(errorMessage);
      }

      const data: ExecutionResult = await response.json();
      return { success: true, data };
    } catch (error) {
      console.error("ApiService: Execution error:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Execution failed"
      };
    }
  }

  async validateCode(code: string): Promise<{ success: boolean; data?: CodeValidationResult; error?: string }> {
    // Validation is handled by the backend during execution
    // Return a simple client-side validation for now
    const isBasicValidation = code.trim().length > 0 && !code.includes('__import__');

    return {
      success: true,
      data: {
        is_safe: isBasicValidation,
        violations: isBasicValidation ? [] : ['Code contains potentially dangerous content'],
        warnings: [],
        sanitized_code: code
      }
    };
  }

  // Enhanced sendMessage method that integrates with the real backend
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      console.log('ApiService: Processing message request:', request);

      // If files are provided, upload them first
      let uploadResults: UploadResult[] = [];
      if (request.files && request.files.length > 0) {
        const uploadResponse = await this.uploadFiles(request.files);
        if (!uploadResponse.success) {
          throw new Error(uploadResponse.error || "Upload failed");
        }
        uploadResults = uploadResponse.data || [];
      }

      // For now, create a response with upload information
      // Later this will be enhanced with AI-generated analysis
      const uploadInfo = uploadResults.length > 0
        ? `\n\n**Data Files Uploaded Successfully:**\n${uploadResults.map(r =>
          `📊 **${r.filename}**: ${r.rows} rows × ${r.columns} columns`
        ).join('\n')}\n\nYour data is ready for analysis! You can now ask questions like:\n• "Show me a summary of the data"\n• "What is the highest salary?"\n• "Create visualizations of the data"\n• "Analyze the relationships between columns"`
        : "";

      return {
        id: Date.now().toString(),
        content: `I've received your message: "${request.message}"${uploadInfo}`,
        timestamp: new Date().toISOString(),
        files: uploadResults.map(r => ({
          id: r.file_id,
          name: r.filename,
          analysisResult: `Data loaded: ${r.rows} rows × ${r.columns} columns`
        }))
      };
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : "Send message failed");
    }
  }

  // Real LLM-powered analysis (replaces fake AI)
  async *streamMessage(request: ChatRequest): AsyncGenerator<string, void, unknown> {
    try {
      console.log('ApiService: Starting REAL LLM analysis with request:', request);

      // Handle file uploads first if present
      if (request.files && request.files.length > 0) {
        yield "📤 **Uploading files...**\n\n";
        await new Promise(resolve => setTimeout(resolve, 500));

        const uploadResponse = await this.uploadFiles(request.files);
        if (uploadResponse.success && uploadResponse.data) {
          for (const result of uploadResponse.data) {
            yield `📊 **${result.filename}** uploaded successfully: ${result.rows} rows × ${result.columns} columns\n\n`;
            await new Promise(resolve => setTimeout(resolve, 300));
          }
          yield "✅ **Files uploaded and ready for analysis!**\n\n";
          await new Promise(resolve => setTimeout(resolve, 500));
        } else {
          yield `❌ **Upload failed:** ${uploadResponse.error}\n\n`;
          return;
        }
      }

      // REAL LLM ANALYSIS - Use the actual backend LLM system
      yield "� **Analyzing your request with AI...**\n\n";
      await new Promise(resolve => setTimeout(resolve, 500));

      try {
        const sessionId = await this.ensureSession();

        // Call the REAL LLM endpoint
        const response = await fetch(`${this.baseUrl}/api/sessions/${sessionId}/analyze-llm?query=${encodeURIComponent(request.message)}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `LLM analysis failed: ${response.statusText}`);
        }

        const analysisResult = await response.json();
        console.log('ApiService: Real LLM analysis result:', analysisResult);

        if (analysisResult.success) {
          // Show AI interpretation
          yield "🤖 **AI Analysis Complete!**\n\n";
          yield `**${analysisResult.interpretation}**\n\n`;

          // Show generated code if available
          if (analysisResult.generated_code) {
            yield "📝 **Generated Analysis Code:**\n";
            if (analysisResult.code_explanation) {
              yield `*${analysisResult.code_explanation}*\n\n`;
            }
            yield "```python\n" + analysisResult.generated_code + "\n```\n\n";
          }

          // Show execution results if available
          if (analysisResult.execution_output) {
            yield "� **Analysis Results:**\n```\n";
            yield analysisResult.execution_output;
            yield "\n```\n\n";
          }

          // Show visualizations if available
          if (analysisResult.visualizations && analysisResult.visualizations.length > 0) {
            yield `📈 **Generated ${analysisResult.visualizations.length} visualization(s)**\n\n`;
          }

          // Show processing metrics
          yield `⚡ **Processing time:** ${analysisResult.processing_time.toFixed(2)} seconds\n`;
          if (analysisResult.retry_count > 0) {
            yield `🔄 **Retries:** ${analysisResult.retry_count}\n`;
          }
          yield "\n";

        } else {
          // Handle analysis failure
          yield "❌ **AI Analysis Failed**\n\n";
          yield analysisResult.interpretation + "\n\n";

          if (analysisResult.execution_error) {
            yield "**Error Details:**\n```\n";
            yield analysisResult.execution_error;
            yield "\n```\n\n";
          }
        }

      } catch (llmError) {
        yield "❌ **LLM Analysis Error:** " + (llmError instanceof Error ? llmError.message : "Unknown error") + "\n\n";
        console.error('ApiService: LLM analysis error:', llmError);
      }

      yield "💡 **What would you like to explore next?**\n";
      yield "• Ask more specific questions about your data\n";
      yield "• Request different types of visualizations\n";
      yield "• Explore relationships between variables\n";
      yield "• Get statistical summaries of specific columns\n";

    } catch (error) {
      yield `❌ **Error:** ${error instanceof Error ? error.message : "Unknown error"}`;
    }
  }

  // Clear session (for page refresh/reset)
  clearSession(): void {
    this.sessionId = null;
  }

  // Get current session ID
  getSessionId(): string | null {
    return this.sessionId;
  }

  // Health check endpoint
  async healthCheck(): Promise<{ success: boolean; status?: string; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.statusText}`);
      }

      const data = await response.json();
      return { success: true, status: data.status };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Health check failed"
      };
    }
  }

  // Legacy method for compatibility  
  async getAnalysisStatus(_fileId: string): Promise<{ status: string; result?: string }> {
    return { status: 'analyzed', result: 'Analysis complete' };
  }
}

export const apiService = new ApiService();
