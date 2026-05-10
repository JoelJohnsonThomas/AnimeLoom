import { cn } from '../utils';

interface LogoMarkProps {
  size?: number;
  withWordmark?: boolean;
  className?: string;
}

/**
 * AnimeLoom brand mark: pink→blue gradient square with stylised inner "AL" glyph.
 * Used in the sidebar header and the Dashboard hero.
 */
export function LogoMark({ size = 32, withWordmark, className }: LogoMarkProps) {
  return (
    <div className={cn('flex items-center gap-2.5', className)}>
      <div
        className="relative flex items-center justify-center rounded-lg shrink-0"
        style={{
          width: size,
          height: size,
          background:
            'linear-gradient(135deg, var(--color-pink) 0%, var(--color-blue) 100%)',
          boxShadow: 'var(--shadow-glow-pink)',
        }}
        aria-hidden
      >
        <svg
          viewBox="0 0 24 24"
          width={size * 0.55}
          height={size * 0.55}
          fill="none"
          stroke="var(--color-fg-inverse)"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M5 19 L9 5 L13 19" />
          <path d="M7 14 L11 14" />
          <path d="M15 19 L15 5" />
          <path d="M15 19 L19 19" />
        </svg>
      </div>
      {withWordmark && (
        <span
          className="text-fg-1 font-display font-bold tracking-tight"
          style={{ fontSize: size * 0.52 }}
        >
          AnimeLoom
        </span>
      )}
    </div>
  );
}
