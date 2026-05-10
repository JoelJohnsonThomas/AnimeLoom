import { useMutation } from '@tanstack/react-query';
import { generateAnime, generateSequence, generateShot } from '../api';
import type {
  SequenceGenerateRequest,
  ShotGenerateRequest,
  TextToAnimeRequest,
} from '../types';

export function useGenerateAnime() {
  return useMutation({ mutationFn: (body: TextToAnimeRequest) => generateAnime(body) });
}

export function useGenerateSequence() {
  return useMutation({
    mutationFn: (body: SequenceGenerateRequest) => generateSequence(body),
  });
}

export function useGenerateShot() {
  return useMutation({ mutationFn: (body: ShotGenerateRequest) => generateShot(body) });
}
