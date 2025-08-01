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
}

interface ExecutionResult {
  result: any;
  code: string;
  output: string;
  error?: string;
  execution_time: number;
  visualizations?: Array<{
    type: string;
    data: string;
    title?: string;
  }>;
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
        throw new Error(`Session creation failed: ${response.statusText}`);
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
  } async uploadFiles(files: File[]): Promise<{ success: boolean; data?: UploadResult[]; error?: string }> {
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
          throw new Error(`Upload failed for ${file.name}: ${response.statusText}`);
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
          code
        }),
      });

      if (!response.ok) {
        throw new Error(`Execution failed: ${response.statusText}`);
      }

      const data: ExecutionResult = await response.json();
      return { success: true, data };
    } catch (error) {
      console.error("Execution error:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Execution failed"
      };
    }
  }

  // Legacy method for compatibility - now uses our backend
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      // If files are provided, upload them first
      let uploadResults: UploadResult[] = [];
      if (request.files && request.files.length > 0) {
        const uploadResponse = await this.uploadFiles(request.files);
        if (!uploadResponse.success) {
          throw new Error(uploadResponse.error || "Upload failed");
        }
        uploadResults = uploadResponse.data || [];
      }

      // Create a response that includes upload information
      const uploadInfo = uploadResults.length > 0
        ? `\n\n**Data Files Uploaded:**\n${uploadResults.map(r =>
          `📊 **${r.filename}**: ${r.rows} rows × ${r.columns} columns`
        ).join('\n')}\n\nYour data is ready for analysis! What would you like to explore?`
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

  // Stream chat responses with code execution
  async *streamMessage(request: ChatRequest): AsyncGenerator<string, void, unknown> {
    try {
      // Handle file uploads first if present
      if (request.files && request.files.length > 0) {
        const uploadResponse = await this.uploadFiles(request.files);
        if (uploadResponse.success && uploadResponse.data) {
          for (const result of uploadResponse.data) {
            yield `📊 **${result.filename}** uploaded: ${result.rows} rows × ${result.columns} columns\n\n`;
            await new Promise(resolve => setTimeout(resolve, 500));
          }
        }
      }

      // Generate Python code based on the user's request
      const message = request.message.toLowerCase();
      let code = "";

      if (message.includes('highest') && message.includes('salary')) {
        code = [
          "# The dataframe 'df' is already provided by the backend",
          "# It contains the uploaded CSV data",
          "",
          "# Find employee with highest salary",
          "highest_salary_idx = df['Salary'].idxmax()",
          "highest_salary_employee = df.loc[highest_salary_idx]",
          "",
          'print("Employee with highest salary:")',
          'print(f"Name: {highest_salary_employee[\'Name\']}")',
          'print(f"Salary: ${highest_salary_employee[\'Salary\']:,}")',
          'print(f"Department: {highest_salary_employee[\'Department\']}")',
          'print(f"Age: {highest_salary_employee[\'Age\']}")',
          'print(f"Experience: {highest_salary_employee[\'Experience\']} years")'
        ].join('\n');

      } else if (message.includes('analyze') || message.includes('summary') || message.includes('describe')) {
        code = [
          "# The dataframe 'df' is already provided by the backend",
          "# It contains the uploaded CSV data",
          "",
          'print("=== DATASET OVERVIEW ===")',
          'print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")',
          'print(f"\\nColumns: {list(df.columns)}")',
          "",
          'print("\\n=== DATA TYPES ===")',
          "print(df.dtypes)",
          "",
          'print("\\n=== MISSING VALUES ===")',
          "missing = df.isnull().sum()",
          "if missing.sum() > 0:",
          "    print(missing[missing > 0])",
          "else:",
          '    print("No missing values found!")',
          "",
          'print("\\n=== SUMMARY STATISTICS ===")',
          "print(df.describe())",
          "",
          "if 'Salary' in df.columns:",
          '    print("\\n=== SALARY ANALYSIS ===")',
          '    print(f"Average Salary: ${df[\'Salary\'].mean():,.2f}")',
          '    print(f"Median Salary: ${df[\'Salary\'].median():,.2f}")',
          '    print(f"Salary Range: ${df[\'Salary\'].min():,} - ${df[\'Salary\'].max():,}")',
          "",
          "if 'Department' in df.columns:",
          '    print("\\n=== DEPARTMENT BREAKDOWN ===")',
          "    dept_counts = df['Department'].value_counts()",
          "    for dept, count in dept_counts.items():",
          "        if 'Salary' in df.columns:",
          "            avg_salary = df[df['Department'] == dept]['Salary'].mean()",
          '            print(f"{dept}: {count} employees (Avg Salary: ${avg_salary:,.2f})")',
          "        else:",
          '            print(f"{dept}: {count} employees")'
        ].join('\n');

      } else if (message.includes('plot') || message.includes('chart') || message.includes('graph') || message.includes('visualiz')) {
        code = [
          "# The dataframe 'df' is already provided by the backend",
          "# Import required libraries for visualization",
          "import matplotlib.pyplot as plt",
          "import seaborn as sns",
          "",
          "# Create visualizations",
          "fig, axes = plt.subplots(2, 2, figsize=(15, 12))",
          "fig.suptitle('Employee Data Analysis', fontsize=16)",
          "",
          "# 1. Salary distribution",
          "if 'Salary' in df.columns:",
          "    axes[0, 0].hist(df['Salary'], bins=10, edgecolor='black', alpha=0.7)",
          "    axes[0, 0].set_title('Salary Distribution')",
          "    axes[0, 0].set_xlabel('Salary')",
          "    axes[0, 0].set_ylabel('Frequency')",
          "",
          "# 2. Department counts",
          "if 'Department' in df.columns:",
          "    dept_counts = df['Department'].value_counts()",
          "    axes[0, 1].bar(dept_counts.index, dept_counts.values)",
          "    axes[0, 1].set_title('Employees by Department')",
          "    axes[0, 1].set_xlabel('Department')",
          "    axes[0, 1].set_ylabel('Number of Employees')",
          "    axes[0, 1].tick_params(axis='x', rotation=45)",
          "",
          "# 3. Age vs Salary scatter plot",
          "if 'Age' in df.columns and 'Salary' in df.columns:",
          "    axes[1, 0].scatter(df['Age'], df['Salary'], alpha=0.6)",
          "    axes[1, 0].set_title('Age vs Salary')",
          "    axes[1, 0].set_xlabel('Age')",
          "    axes[1, 0].set_ylabel('Salary')",
          "",
          "# 4. Experience vs Salary",
          "if 'Experience' in df.columns and 'Salary' in df.columns:",
          "    axes[1, 1].scatter(df['Experience'], df['Salary'], alpha=0.6)",
          "    axes[1, 1].set_title('Experience vs Salary')",
          "    axes[1, 1].set_xlabel('Years of Experience')",
          "    axes[1, 1].set_ylabel('Salary')",
          "",
          "plt.tight_layout()",
          "plt.show()",
          "",
          'print("Visualizations created successfully!")'
        ].join('\n');

      } else {
        // Generic exploration code
        code = [
          "# The dataframe 'df' is already provided by the backend",
          "# It contains the uploaded CSV data",
          "",
          'print("=== DATA PREVIEW ===")',
          'print("First 5 rows:")',
          "print(df.head())",
          "",
          'print("\\n=== BASIC INFO ===")',
          'print(f"Dataset shape: {df.shape}")',
          'print(f"Columns: {list(df.columns)}")',
          "",
          'print("\\nWhat would you like to explore?")',
          'print("• Ask about specific columns or relationships")',
          'print("• Request visualizations or charts")',
          'print("• Ask for summary statistics")',
          'print("• Find specific information (e.g., \'highest salary\', \'average age\')")'
        ].join('\n');
      }

      // Show the code that will be executed
      yield "```python\n" + code + "\n```\n\n";
      await new Promise(resolve => setTimeout(resolve, 500));

      // Execute the code using the backend
      yield "**Executing code...**\n\n";

      try {
        const executionResponse = await this.executeCode(code);

        if (executionResponse.success && executionResponse.data) {
          yield "**Results:**\n```\n";
          yield executionResponse.data.output || "Code executed successfully";
          yield "\n```\n\n";

          if (executionResponse.data.error) {
            yield "**Note:** " + executionResponse.data.error + "\n\n";
          }
        } else {
          yield "**Error:** " + (executionResponse.error || "Code execution failed") + "\n\n";
        }
      } catch (execError) {
        yield "**Execution Error:** " + (execError instanceof Error ? execError.message : "Unknown error") + "\n\n";
      }

      yield "Would you like me to explore any other aspects of your data?\n";

    } catch (error) {
      yield `Error: ${error instanceof Error ? error.message : "Unknown error"}`;
    }
  }  // Clear session (for page refresh/reset)
  clearSession(): void {
    this.sessionId = null;
  }

  // Get current session ID
  getSessionId(): string | null {
    return this.sessionId;
  }

  // Legacy methods for compatibility
  async getAnalysisStatus(fileId: string): Promise<{ status: string; result?: string }> {
    return { status: 'analyzed', result: 'Analysis complete' };
  }
}

export const apiService = new ApiService();
