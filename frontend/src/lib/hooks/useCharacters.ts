import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  listCharacters,
  createCharacter,
  deleteCharacter,
  getCharacter,
} from '../api';
import type { CharacterCreateRequest } from '../types';

export const CHARACTER_KEYS = {
  all: ['characters'] as const,
  detail: (id: string) => ['character', id] as const,
};

export function useCharacters() {
  return useQuery({
    queryKey: CHARACTER_KEYS.all,
    queryFn: listCharacters,
    refetchInterval: (q) => {
      const data = q.state.data;
      // Poll while any character is still training (has_lora=false).
      const training = data?.characters.some((c) => !c.has_lora);
      return training ? 3000 : false;
    },
  });
}

export function useCharacter(id: string | undefined) {
  return useQuery({
    queryKey: id ? CHARACTER_KEYS.detail(id) : ['character', 'none'],
    queryFn: () => getCharacter(id!),
    enabled: Boolean(id),
  });
}

export function useCreateCharacter() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CharacterCreateRequest) => createCharacter(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: CHARACTER_KEYS.all }),
  });
}

export function useDeleteCharacter() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteCharacter(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: CHARACTER_KEYS.all }),
  });
}
