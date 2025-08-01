import { useState } from 'react';
import { ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';
import { Button } from './button';

interface CodeSnippetProps {
  code: string;
  language?: string;
  title?: string;
  collapsed?: boolean;
}

export function CodeSnippet({ code, language = 'python', title, collapsed = true }: CodeSnippetProps) {
  const [isCollapsed, setIsCollapsed] = useState(collapsed);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  return (
    <div className="border border-border rounded-lg overflow-hidden bg-card">
      {/* Header */}
      <div className="flex items-center justify-between p-3 bg-muted/50 border-b border-border">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="flex items-center gap-2 text-sm text-foreground hover:text-primary transition-colors"
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
          <span className="font-medium">
            {title || `${language.charAt(0).toUpperCase() + language.slice(1)} Code`}
          </span>
        </button>

        {!isCollapsed && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="text-muted-foreground hover:text-foreground"
          >
            {copied ? (
              <Check className="w-4 h-4" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
        )}
      </div>

      {/* Code Content */}
      {!isCollapsed && (
        <div className="relative">
          <pre className="p-4 text-sm overflow-x-auto bg-background">
            <code className={`language-${language}`}>
              {code}
            </code>
          </pre>
        </div>
      )}
    </div>
  );
}
