import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '../utils';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  iconLeft?: ReactNode;
  iconRight?: ReactNode;
  glow?: boolean;
}

const VARIANT: Record<Variant, string> = {
  primary:
    'bg-pink text-fg-inverse font-semibold hover:bg-pink-dim active:scale-[0.97] transition',
  secondary:
    'bg-transparent text-fg-2 border border-border-subtle hover:text-fg-1 hover:border-border-strong active:scale-[0.97] transition',
  ghost:
    'bg-transparent text-fg-2 hover:text-fg-1 hover:bg-bg-raised active:scale-[0.97] transition',
  danger:
    'bg-red text-fg-inverse font-semibold hover:opacity-90 active:scale-[0.97] transition',
};

const SIZE: Record<Size, string> = {
  sm: 'h-8 px-3 text-[12px] gap-1.5 rounded-md',
  md: 'h-10 px-5 text-[14px] gap-2 rounded-md',
  lg: 'h-12 px-6 text-[15px] gap-2 rounded-lg',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'primary', size = 'md', iconLeft, iconRight, glow, className, children, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center select-none',
        'duration-[120ms]',
        VARIANT[variant],
        SIZE[size],
        glow && variant === 'primary' && 'shadow-[var(--shadow-glow-pink)]',
        className,
      )}
      {...rest}
    >
      {iconLeft}
      {children}
      {iconRight}
    </button>
  );
});
