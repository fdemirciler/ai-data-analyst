import { useState, useCallback } from 'react';
import { apiService } from '../services/api';

export interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

export interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
  execute: (...args: any[]) => Promise<T>;
  reset: () => void;
}

interface UploadedFile {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
  analysisResult?: string;
  rows?: number;
  columns?: number;
}

interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  files?: UploadedFile[];
  isStreaming?: boolean;
  codeSnippets?: Array<{
    code: string;
    language: string;
    collapsed: boolean;
  }>;
}

export function useApi<T>(
  apiFunction: (...args: any[]) => Promise<T>
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const execute = useCallback(async (...args: any[]): Promise<T> => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiFunction(...args);
      setData(result);
      return result;
    } catch (err) {
      const apiError: ApiError = {
        message: err instanceof Error ? err.message : 'An unknown error occurred',
        details: err,
      };
      setError(apiError);
      throw apiError;
    } finally {
      setLoading(false);
    }
  }, [apiFunction]);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { data, loading, error, execute, reset };
}

export const useAnalyticsApi = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uploadFiles = useCallback(async (files: File[]): Promise<UploadedFile[]> => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await apiService.uploadFiles(files);

      if (!response.success) {
        throw new Error(response.error || 'Upload failed');
      }

      // Convert API response to UI format
      console.log('useAnalyticsApi: Raw API response:', response);
      const uploadedFiles: UploadedFile[] = response.data?.map((result, index) => {
        console.log('useAnalyticsApi: Processing result:', result);
        return {
          id: result.file_id || `file_${Date.now()}_${index}`,
          name: result.filename,
          type: files[index]?.type || 'application/octet-stream',
          size: files[index]?.size || 0,
          url: files[index] ? URL.createObjectURL(files[index]) : '',
          analysisResult: `Data loaded: ${result.rows} rows × ${result.columns} columns`,
          rows: result.rows,
          columns: result.columns
        };
      }) || [];

      console.log('useAnalyticsApi: Converted uploaded files:', uploadedFiles);
      return uploadedFiles;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []); const sendMessage = useCallback(async (
    message: string,
    files?: File[]
  ): Promise<{ response: string; uploadedFiles?: UploadedFile[] }> => {
    setIsLoading(true);
    setError(null);

    try {
      let uploadedFiles: UploadedFile[] = [];

      // Upload files if provided
      if (files && files.length > 0) {
        uploadedFiles = await uploadFiles(files);
      }

      // For now, return a simple response
      // In a real implementation, this would call an AI service
      let response = "";
      if (uploadedFiles.length > 0) {
        response = `I've successfully uploaded your data files:\n\n${uploadedFiles.map(f =>
          `📊 **${f.name}**: ${f.rows} rows × ${f.columns} columns`
        ).join('\n')}\n\nYour data is ready for analysis! What would you like to explore?`;
      } else {
        response = "I understand you'd like to work with your data. Please upload a CSV or Excel file, and I can help you analyze it!";
      }

      return { response, uploadedFiles };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Request failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [uploadFiles]);

  const streamMessage = useCallback(async function* (
    message: string,
    files?: File[]
  ): AsyncGenerator<{ content: string; isComplete: boolean; uploadedFiles?: UploadedFile[] }, void, unknown> {
    setIsLoading(true);
    setError(null);

    try {
      let uploadedFiles: UploadedFile[] = [];

      // Upload files if provided
      if (files && files.length > 0) {
        uploadedFiles = await uploadFiles(files);
        yield {
          content: uploadedFiles.map(f =>
            `📊 **${f.name}** uploaded: ${f.rows} rows × ${f.columns} columns`
          ).join('\n') + '\n\n',
          isComplete: false,
          uploadedFiles
        };
      }

      // Stream the AI response
      const request = { message, files };
      const stream = apiService.streamMessage(request);

      for await (const chunk of stream) {
        yield { content: chunk, isComplete: false };
      }

      yield { content: '', isComplete: true };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Stream failed';
      setError(errorMessage);
      yield { content: `Error: ${errorMessage}`, isComplete: true };
    } finally {
      setIsLoading(false);
    }
  }, [uploadFiles]);

  const clearSession = useCallback(() => {
    apiService.clearSession();
  }, []);

  return {
    uploadFiles,
    sendMessage,
    streamMessage,
    clearSession,
    isLoading,
    error,
    clearError: () => setError(null)
  };
};

export function useFileUpload() {
  return useApi(async (files: File[]) => {
    return apiService.uploadFiles(files);
  });
}

export function useChat() {
  return useApi(async (message: string, files?: File[], sessionId?: string) => {
    return apiService.sendMessage({ message, files, sessionId });
  });
}
