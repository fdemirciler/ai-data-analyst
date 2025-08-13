interface ContentRendererProps {
  content: string;
  className?: string;
}

export function ContentRenderer({ content, className = "" }: ContentRendererProps) {
  // Check if content contains HTML tables
  const hasHTMLTables = content.includes('<table') && content.includes('</table>');

  if (!hasHTMLTables) {
    // Render as plain text with whitespace preservation
    return (
      <div className={`whitespace-pre-wrap leading-relaxed ${className}`}>
        {content}
      </div>
    );
  }

  // Sanitize and render HTML content
  const sanitizedHTML = sanitizeHTML(content);

  return (
    <div
      className={`leading-relaxed content-renderer ${className}`}
      dangerouslySetInnerHTML={{ __html: sanitizedHTML }}
    />
  );
}

// Basic HTML sanitization for table content
function sanitizeHTML(html: string): string {
  let sanitized = html;

  // Remove script tags completely
  sanitized = sanitized.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');

  // Remove any HTML tags not in our allowed list (basic protection)
  sanitized = sanitized.replace(/<(?!\/?(?:table|thead|tbody|tr|th|td|p|br|strong|em|b|i)\b)[^>]*>/gi, '');

  // Remove any potential JavaScript event handlers
  sanitized = sanitized.replace(/\s*on\w+\s*=\s*[^>]*/gi, '');

  // Ensure table has our CSS class
  sanitized = sanitized.replace(
    /<table([^>]*)>/gi,
    '<table class="analysis-table" $1>'
  );

  return sanitized;
}

export default ContentRenderer;
