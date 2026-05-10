import { create } from 'zustand';

interface AppState {
  activeJobId: string | null;
  setActiveJob: (id: string | null) => void;

  /** Toasts: ephemeral notifications fired by job completion etc. */
  toasts: Toast[];
  pushToast: (toast: Omit<Toast, 'id'>) => void;
  dismissToast: (id: string) => void;
}

export interface Toast {
  id: string;
  kind: 'info' | 'success' | 'error';
  title: string;
  body?: string;
}

export const useAppStore = create<AppState>((set) => ({
  activeJobId: null,
  setActiveJob: (id) => set({ activeJobId: id }),

  toasts: [],
  pushToast: (t) =>
    set((s) => ({
      toasts: [...s.toasts, { ...t, id: crypto.randomUUID() }],
    })),
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));
