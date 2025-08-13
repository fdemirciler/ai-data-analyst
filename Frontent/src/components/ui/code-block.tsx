import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check, Code, ChevronDown, ChevronRight } from 'lucide-react';
import { Button } from './button';
import { Collapsible, CollapsibleContent } from './collapsible';

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
  showLineNumbers?: boolean;
}

export function CodeBlock({
  code,
  language = 'python',
  title = 'Generated Python Script',
  showLineNumbers = true
}: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false);
  const [isOpen, setIsOpen] = React.useState(true);

  // Check if we're in dark mode
  const [isDarkMode, setIsDarkMode] = React.useState(false);

  React.useEffect(() => {
    // Check initial dark mode state
    const checkDarkMode = () => {
      const isDark = document.documentElement.classList.contains('dark');
      setIsDarkMode(isDark);
    };

    checkDarkMode();

    // Watch for dark mode changes
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  if (!code || !code.trim()) {
    return null;
  }

  const syntaxTheme = isDarkMode ? oneDark : oneLight;

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="bg-card rounded-lg border overflow-hidden">
        {/* Header */}
        <div
          className="flex items-center justify-between px-4 py-2 border-b bg-muted/50 hover:bg-muted/70 cursor-pointer transition-colors"
          onClick={() => setIsOpen(!isOpen)}
        >
          <div className="flex items-center gap-2">
            {isOpen ? (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            )}
            <Code className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium">{title}</span>
            <span className="text-xs text-muted-foreground bg-blue-100 dark:bg-blue-900 px-2 py-1 rounded">
              {language}
            </span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              handleCopy();
            }}
            className="h-8 w-8 p-0"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
        </div>

        {/* Code Content */}
        {isOpen && (
          <CollapsibleContent>
            <div className="relative">
              <SyntaxHighlighter
                language={language}
                style={syntaxTheme}
                showLineNumbers={showLineNumbers}
                customStyle={{
                  margin: 0,
                  padding: '1rem',
                  background: 'transparent',
                  fontSize: '0.875rem',
                  lineHeight: '1.5',
                }}
                lineNumberStyle={{
                  fontSize: '0.75rem',
                  paddingRight: '1rem',
                  userSelect: 'none',
                  opacity: 0.6,
                }}
                wrapLines={true}
                wrapLongLines={true}
              >
                {code}
              </SyntaxHighlighter>
            </div>
          </CollapsibleContent>
        )}
      </div>
    </Collapsible>
  );
}

export default CodeBlock;
