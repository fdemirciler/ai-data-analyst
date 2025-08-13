"use client";

import React from "react";

interface CollapsibleProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: React.ReactNode;
}

function Collapsible({ open, onOpenChange, children, ...props }: CollapsibleProps) {
  return <div data-slot="collapsible" {...props}>{children}</div>;
}

interface CollapsibleTriggerProps {
  asChild?: boolean;
  children: React.ReactNode;
  onClick?: (e: React.MouseEvent) => void;
}

function CollapsibleTrigger({ asChild, children, onClick, ...props }: CollapsibleTriggerProps) {
  if (asChild) {
    return React.cloneElement(children as React.ReactElement, {
      onClick,
      ...props,
    });
  }
  return (
    <button onClick={onClick} data-slot="collapsible-trigger" {...props}>
      {children}
    </button>
  );
}

interface CollapsibleContentProps {
  children: React.ReactNode;
}

function CollapsibleContent({ children, ...props }: CollapsibleContentProps) {
  return (
    <div data-slot="collapsible-content" {...props}>
      {children}
    </div>
  );
}

export { Collapsible, CollapsibleTrigger, CollapsibleContent };
