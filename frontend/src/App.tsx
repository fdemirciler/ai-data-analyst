import { useState, useRef, useEffect } from "react";
import {
  RefreshCw,
  ArrowUp,
  Paperclip,
  Sun,
  Moon,
  X,
  FileText,
  Image as ImageIcon,
} from "lucide-react";
import { Button } from "./components/ui/button";
import { Textarea } from "./components/ui/textarea";
import { CodeSnippet } from "./components/ui/code-snippet";
import { useAnalyticsApi } from "./hooks/useApi";

// Agent Logo Component - atomic orbital design
function BoxAILogo({
  size = "md",
}: {
  size?: "sm" | "md" | "lg";
}) {
  const sizeClasses = {
    sm: "w-6 h-6",
    md: "w-8 h-8",
    lg: "w-12 h-12",
  };

  return (
    <div
      className={`relative ${sizeClasses[size]} flex items-center justify-center`}
    >
      <svg viewBox="0 0 48 48" className="w-full h-full">
        {/* Central core */}
        <circle cx="24" cy="24" r="3" fill="#e91e63" />

        {/* Orbital rings */}
        <ellipse
          cx="24"
          cy="24"
          rx="18"
          ry="8"
          fill="none"
          stroke="#e91e63"
          strokeWidth="2"
          transform="rotate(0 24 24)"
          opacity="0.8"
        />
        <ellipse
          cx="24"
          cy="24"
          rx="18"
          ry="8"
          fill="none"
          stroke="#2196f3"
          strokeWidth="2"
          transform="rotate(60 24 24)"
          opacity="0.8"
        />
        <ellipse
          cx="24"
          cy="24"
          rx="18"
          ry="8"
          fill="none"
          stroke="#9c27b0"
          strokeWidth="2"
          transform="rotate(120 24 24)"
          opacity="0.8"
        />

        {/* Orbital particles */}
        <circle cx="42" cy="24" r="2" fill="#e91e63" />
        <circle cx="33" cy="39" r="2" fill="#2196f3" />
        <circle cx="15" cy="39" r="2" fill="#9c27b0" />
      </svg>
    </div>
  );
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
    title?: string;
    collapsed: boolean;
  }>;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isInputExpanded, setIsInputExpanded] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const api = useAnalyticsApi();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    console.log('Dark mode effect triggered, isDarkMode:', isDarkMode);
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
      console.log('Added dark class to html element');
    } else {
      document.documentElement.classList.remove("dark");
      console.log('Removed dark class from html element');
    }
    // Store preference
    localStorage.setItem('darkMode', JSON.stringify(isDarkMode));
  }, [isDarkMode]);

  const toggleDarkMode = () => {
    console.log('Dark mode toggle clicked, current state:', isDarkMode);
    setIsDarkMode(!isDarkMode);
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    console.log('App: File change event triggered');
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;

    console.log('App: Selected files:', files.map(f => f.name));

    // Create temporary file objects for UI
    const tempFiles: UploadedFile[] = files.map(file => ({
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      name: file.name,
      type: file.type,
      size: file.size,
      url: URL.createObjectURL(file),
      analysisResult: "Processing..."
    }));

    console.log('App: Created temp files:', tempFiles);
    setUploadedFiles(prev => [...prev, ...tempFiles]);

    try {
      // Upload files using the API
      console.log('App: Starting API upload...');
      const uploadedFiles = await api.uploadFiles(files);
      console.log('App: API upload completed:', uploadedFiles);

      // Update the temporary files with actual results
      setUploadedFiles(prev => prev.map(tempFile => {
        const uploaded = uploadedFiles.find(u => u.name === tempFile.name);
        if (uploaded) {
          return {
            ...tempFile,
            id: uploaded.id,
            analysisResult: uploaded.analysisResult || `Data loaded: ${uploaded.rows} rows × ${uploaded.columns} columns`,
            rows: uploaded.rows,
            columns: uploaded.columns
          };
        }
        return tempFile;
      }));
    } catch (error) {
      console.error('App: Upload failed:', error);
      // Update failed files
      setUploadedFiles(prev => prev.map(tempFile => {
        if (tempFiles.some(t => t.id === tempFile.id)) {
          return { ...tempFile, analysisResult: "Upload failed" };
        }
        return tempFile;
      }));
    }

    // Clear the input
    if (event.target) {
      event.target.value = "";
    }
  }; const removeFile = (fileId: string) => {
    setUploadedFiles((prev) => {
      const fileToRemove = prev.find((f) => f.id === fileId);
      if (fileToRemove) {
        URL.revokeObjectURL(fileToRemove.url);
      }
      return prev.filter((f) => f.id !== fileId);
    });
  };

  const handleSendMessage = async () => {
    if ((!inputText.trim() && uploadedFiles.length === 0) || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputText.trim(),
      timestamp: new Date(),
      files: uploadedFiles.length > 0 ? [...uploadedFiles] : undefined,
    };

    setMessages(prev => [...prev, userMessage]);
    const originalFiles = [...uploadedFiles];
    setInputText("");
    setUploadedFiles([]);
    setIsLoading(true);

    // Create assistant message for streaming
    const assistantMessageId = (Date.now() + 1).toString();
    setStreamingMessageId(assistantMessageId);

    const assistantMessage: Message = {
      id: assistantMessageId,
      type: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
      codeSnippets: []
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      // Convert uploaded files to File objects for API
      const filesToSend = originalFiles.length > 0
        ? await Promise.all(originalFiles.map(async (uploadedFile) => {
          const response = await fetch(uploadedFile.url);
          const blob = await response.blob();
          return new File([blob], uploadedFile.name, { type: uploadedFile.type });
        }))
        : undefined;

      // Stream the response
      const stream = api.streamMessage(userMessage.content, filesToSend);
      let accumulatedContent = "";
      let currentCodeSnippets: Array<{ code: string; language: string; title?: string; collapsed: boolean }> = [];

      for await (const chunk of stream) {
        if (chunk.isComplete) break;

        accumulatedContent += chunk.content;

        // Parse code snippets from content
        const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
        const matches = [...accumulatedContent.matchAll(codeBlockRegex)];

        currentCodeSnippets = matches.map((match, index) => ({
          code: match[2].trim(),
          language: match[1] || 'text',
          title: `Code Block ${index + 1}`,
          collapsed: true
        }));

        // Remove code blocks from display content
        const contentWithoutCode = accumulatedContent.replace(codeBlockRegex, '[Code snippet extracted]');

        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId
            ? {
              ...msg,
              content: contentWithoutCode,
              codeSnippets: currentCodeSnippets
            }
            : msg
        ));
      }

      // Mark streaming as complete
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMessageId
          ? { ...msg, isStreaming: false }
          : msg
      ));

    } catch (error) {
      console.error('Send message error:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMessageId
          ? {
            ...msg,
            content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
            isStreaming: false
          }
          : msg
      ));
    } finally {
      setIsLoading(false);
      setStreamingMessageId(null);
    }
  };

  const handleRefresh = () => {
    setMessages([]);
    setInputText("Upload your file (csv/xlsx) and ask a question about your data...");
    setUploadedFiles([]);
    api.clearSession(); // Clear the backend session
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInputFocus = () => {
    setIsInputExpanded(true);
  };

  const handleInputBlur = () => {
    if (!inputText.trim() && uploadedFiles.length === 0) {
      setIsInputExpanded(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    if (e.target.value.trim() || uploadedFiles.length > 0) {
      setIsInputExpanded(true);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (
      parseFloat((bytes / Math.pow(k, i)).toFixed(2)) +
      " " +
      sizes[i]
    );
  };

  const hasContent =
    inputText.trim() || uploadedFiles.length > 0;

  return (
    <>
      <style>{`
        @keyframes subtle-glow {
          0%, 100% { 
            box-shadow: 0 0 4px rgba(168, 85, 247, 0.15), 0 0 8px rgba(168, 85, 247, 0.1);
          }
          50% { 
            box-shadow: 0 0 8px rgba(168, 85, 247, 0.25), 0 0 16px rgba(168, 85, 247, 0.15);
          }
        }
        .glow-border {
          animation: subtle-glow 8s ease-in-out infinite;
        }
        .glow-button {
          animation: subtle-glow 8s ease-in-out infinite;
        }
        .hide-scrollbar {
          scrollbar-width: none; /* Firefox */
          -ms-overflow-style: none; /* Internet Explorer 10+ */
        }
        .hide-scrollbar::-webkit-scrollbar {
          width: 0;
          height: 0;
          display: none; /* WebKit */
        }
      `}</style>

      <div className="h-screen bg-background flex flex-col transition-colors duration-300">
        {/* Header */}
        <header className="bg-card border-b border-border px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <BoxAILogo size="md" />
            <h1 className="text-xl font-semibold bg-gradient-to-r from-pink-500 via-blue-500 to-purple-600 bg-clip-text text-transparent">AI Analyst Agent</h1>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground hover:text-foreground"
              onClick={handleRefresh}
            >
              <RefreshCw className="w-5 h-5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground hover:text-foreground"
              onClick={toggleDarkMode}
            >
              {isDarkMode ? (
                <Sun className="w-5 h-5" />
              ) : (
                <Moon className="w-5 h-5" />
              )}
            </Button>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {messages.length === 0 ? (
            /* Welcome Screen */
            <div className="flex-1 flex flex-col items-center justify-center px-6 pb-32">
              <div className="text-center space-y-6 max-w-2xl">
                <div className="space-y-2">
                  <p className="text-xl text-muted-foreground">
                    AI powered data analysis
                  </p>
                </div>

                <p className="text-muted-foreground mt-8">
                  Chat cleared when you refresh the page
                </p>
              </div>
            </div>
          ) : (
            /* Chat Interface */
            <div className="flex-1 flex flex-col overflow-hidden">
              <div className="flex-1 overflow-y-auto px-6 py-6">
                <div className="max-w-4xl mx-auto space-y-6">
                  {messages.map((message) => (
                    <div
                      key={message.id}
                      className="flex gap-4"
                    >
                      <div className="flex-shrink-0">
                        {message.type === "user" ? (
                          <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm">
                            U
                          </div>
                        ) : (
                          <BoxAILogo size="sm" />
                        )}
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="text-sm text-muted-foreground">
                          {message.type === "user"
                            ? "You"
                            : "Agent"}{" "}
                          •{" "}
                          {message.timestamp.toLocaleTimeString()}
                        </div>

                        {/* File attachments */}
                        {message.files &&
                          message.files.length > 0 && (
                            <div className="space-y-2 mb-3">
                              {message.files.map((file) => (
                                <div
                                  key={file.id}
                                  className="bg-muted rounded-lg p-3 border"
                                >
                                  <div className="flex items-center gap-2 mb-2">
                                    {file.type.startsWith("image/") ? (
                                      <ImageIcon className="w-4 h-4 text-blue-500" />
                                    ) : (
                                      <FileText className="w-4 h-4 text-green-500" />
                                    )}
                                    <span className="text-sm font-medium">
                                      {file.name}
                                    </span>
                                    <span className="text-xs text-muted-foreground">
                                      ({formatFileSize(file.size)})
                                    </span>
                                    {file.rows && file.columns && (
                                      <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-2 py-1 rounded">
                                        {file.rows} × {file.columns}
                                      </span>
                                    )}
                                  </div>
                                  {file.type.startsWith(
                                    "image/",
                                  ) && (
                                      <img
                                        src={file.url}
                                        alt={file.name}
                                        className="max-w-xs max-h-48 rounded border object-contain mb-2"
                                      />
                                    )}
                                  {file.analysisResult && (
                                    <div className="text-xs text-muted-foreground mt-2 p-2 bg-background rounded border-l-2 border-blue-500">
                                      {file.analysisResult}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}

                        <div className="text-foreground whitespace-pre-wrap leading-relaxed">
                          {message.content}
                        </div>

                        {/* Code snippets */}
                        {message.codeSnippets && message.codeSnippets.length > 0 && (
                          <div className="space-y-3 mt-3">
                            {message.codeSnippets.map((snippet, index) => (
                              <CodeSnippet
                                key={index}
                                code={snippet.code}
                                language={snippet.language}
                                title={snippet.title}
                                collapsed={snippet.collapsed}
                              />
                            ))}
                          </div>
                        )}

                        {/* Streaming indicator */}
                        {message.isStreaming && (
                          <div className="flex items-center gap-2 mt-2 text-muted-foreground">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                              <div
                                className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                                style={{ animationDelay: "0.1s" }}
                              ></div>
                              <div
                                className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                                style={{ animationDelay: "0.2s" }}
                              ></div>
                            </div>
                            <span className="text-sm">Thinking...</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}

                  {isLoading && (
                    <div className="flex gap-4">
                      <div className="flex-shrink-0">
                        <BoxAILogo size="sm" />
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="text-sm text-muted-foreground">
                          Agent • Thinking...
                        </div>
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                          <div
                            className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                            style={{ animationDelay: "0.1s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Input Section */}
        <div className="px-6 pb-6">
          <div className="max-w-4xl mx-auto">
            {/* File upload previews */}
            {uploadedFiles.length > 0 && (
              <div className="mb-4 space-y-2">
                {uploadedFiles.map((file) => (
                  <div
                    key={file.id}
                    className="bg-muted rounded-lg p-3 border flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2">
                      {file.type.startsWith("image/") ? (
                        <ImageIcon className="w-4 h-4 text-blue-500" />
                      ) : (
                        <FileText className="w-4 h-4 text-green-500" />
                      )}
                      <span className="text-sm">
                        {file.name}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        ({formatFileSize(file.size)})
                      </span>
                      {file.rows && file.columns && (
                        <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-2 py-1 rounded">
                          {file.rows} × {file.columns}
                        </span>
                      )}
                      {file.analysisResult && (
                        <span className="text-xs bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 px-2 py-1 rounded">
                          Analyzed
                        </span>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(file.id)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            <div className="relative">
              <div
                className={`relative rounded-2xl border-2 ${hasContent ? "border-purple-400" : "border-muted"} bg-card p-4 transition-all duration-300 ${isInputExpanded ? 'min-h-32' : 'min-h-16'}`}
              >
                <div className="space-y-3">
                  <Textarea
                    ref={textareaRef}
                    value={inputText}
                    onChange={handleInputChange}
                    onFocus={handleInputFocus}
                    onBlur={handleInputBlur}
                    onKeyPress={handleKeyPress}
                    placeholder="Upload your file (csv/xlsx) and ask a question about your data..."
                    className={`resize-none border border-border/5 rounded-md bg-transparent p-2 text-foreground placeholder:text-muted-foreground focus:ring-0 focus:outline-none focus:border-border/1 transition-all duration-300 hide-scrollbar ${isInputExpanded
                      ? 'min-h-20 max-h-64'
                      : 'min-h-[20px] max-h-8'
                      }`}
                    rows={isInputExpanded ? 3 : 1}
                  />

                  <div className="flex items-center justify-between">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleFileUpload}
                      className="flex items-center gap-2 text-gray-400 hover:text-foreground px-2"
                    >
                      <Paperclip className="w-4 h-4" />
                    </Button>

                    <Button
                      size="icon"
                      onClick={handleSendMessage}
                      disabled={!hasContent || isLoading}
                      className={`rounded-full w-10 h-10 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white border-0 disabled:opacity-50 shadow-lg`}
                    >
                      <ArrowUp className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".csv,.xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,text/csv"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
    </>
  );
}