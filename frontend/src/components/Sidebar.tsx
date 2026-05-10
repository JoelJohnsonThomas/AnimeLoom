import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  UserCircle2,
  FileText,
  Wand2,
  Film,
  Settings as SettingsIcon,
  BookOpen,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { LogoMark } from './brand/LogoMark';
import { cn } from './utils';

interface NavItemDef {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

const NAV: NavItemDef[] = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard, end: true },
  { to: '/characters', label: 'Characters', icon: UserCircle2 },
  { to: '/script', label: 'Script Editor', icon: FileText },
  { to: '/generate', label: 'Generate', icon: Wand2 },
  { to: '/results', label: 'Results', icon: Film },
];

const FOOTER: NavItemDef[] = [
  { to: '/settings', label: 'Settings', icon: SettingsIcon },
  { to: '/docs', label: 'Docs', icon: BookOpen },
];

function Item({ item }: { item: NavItemDef }) {
  const Icon = item.icon;
  return (
    <NavLink
      to={item.to}
      end={item.end}
      className={({ isActive }) =>
        cn(
          'flex items-center gap-2.5 px-2.5 py-2 rounded-md text-[13px] transition-all duration-150',
          'border',
          isActive
            ? 'bg-pink/12 border-pink/20 text-pink font-semibold'
            : 'border-transparent text-fg-2 hover:text-fg-1 hover:bg-bg-deep',
        )
      }
    >
      <Icon size={16} strokeWidth={1.75} className="shrink-0" />
      <span>{item.label}</span>
    </NavLink>
  );
}

export function Sidebar() {
  return (
    <aside className="w-[220px] min-h-screen bg-bg-deep border-r border-border-faint flex flex-col shrink-0">
      <div className="px-5 py-4 border-b border-border-faint">
        <LogoMark size={30} withWordmark />
      </div>

      <nav className="flex-1 p-2.5 flex flex-col gap-0.5">
        {NAV.map((item) => (
          <Item key={item.to} item={item} />
        ))}
      </nav>

      <div className="p-2.5 border-t border-border-faint flex flex-col gap-0.5">
        {FOOTER.map((item) => (
          <Item key={item.to} item={item} />
        ))}
        <span className="px-2.5 pt-2 pb-1 text-[10px] font-mono text-fg-3">v0.1.0</span>
      </div>
    </aside>
  );
}
