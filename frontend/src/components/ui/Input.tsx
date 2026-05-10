import { forwardRef } from 'react';
import type { InputHTMLAttributes, TextareaHTMLAttributes, SelectHTMLAttributes, ReactNode } from 'react';
import { cn } from '../utils';

const base =
  'w-full bg-bg-deep border border-border-subtle rounded-md text-fg-1 placeholder:text-fg-3 ' +
  'transition-colors duration-150 ' +
  'hover:border-border-strong ' +
  'focus:border-pink focus:outline-none ' +
  'disabled:opacity-50 disabled:cursor-not-allowed';

const sizing = 'px-3 py-2 text-[14px]';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  invalid?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { invalid, className, ...rest },
  ref,
) {
  return (
    <input
      ref={ref}
      className={cn(base, sizing, invalid && 'border-red focus:border-red', className)}
      {...rest}
    />
  );
});

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  invalid?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  { invalid, className, ...rest },
  ref,
) {
  return (
    <textarea
      ref={ref}
      className={cn(
        base,
        sizing,
        'font-mono leading-relaxed resize-y',
        invalid && 'border-red focus:border-red',
        className,
      )}
      {...rest}
    />
  );
});

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  children?: ReactNode;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { className, children, ...rest },
  ref,
) {
  return (
    <select ref={ref} className={cn(base, sizing, 'appearance-none pr-8', className)} {...rest}>
      {children}
    </select>
  );
});

interface FieldProps {
  label: string;
  hint?: string;
  error?: string;
  htmlFor?: string;
  children: ReactNode;
}

export function Field({ label, hint, error, htmlFor, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={htmlFor}
        className="text-[11px] font-semibold uppercase tracking-wider text-fg-hint"
      >
        {label}
      </label>
      {children}
      {error ? (
        <span className="text-[12px] text-red">{error}</span>
      ) : hint ? (
        <span className="text-[12px] text-fg-hint">{hint}</span>
      ) : null}
    </div>
  );
}
