/**
 * Tiny className composer. Filters falsy values, joins with space.
 * Avoids clsx/tailwind-merge dependency.
 */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ');
}
