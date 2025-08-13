import { useState, useCallback } from 'react';

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

export function useFileUpload() {
  return useApi(async (files: File[]) => {
    const { apiService } = await import('../services/api');
    if (!files || files.length === 0) throw new Error('No file provided');
    return apiService.uploadFile(files[0]);
  });
}

export function useChat() {
  return useApi(async (sessionId: string, message: string) => {
    const { apiService } = await import('../services/api');
    return apiService.sendMessage({ sessionId, message });
  });
}
