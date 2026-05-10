import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '../utils';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  raised?: boolean;
  hover?: boolean;
  padded?: boolean;
  children?: ReactNode;
}

export function Card({ raised, hover, padded = true, className, children, ...rest }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-lg border transition',
        raised
          ? 'bg-bg-raised border-border-subtle'
          : 'bg-bg-surface border-border-subtle',
        hover && 'hover:bg-bg-raised hover:border-border-strong cursor-pointer',
        padded && 'p-5',
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
