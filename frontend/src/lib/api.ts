import type {
  CharacterCreateRequest,
  CharacterListResponse,
  CharacterResponse,
  JobStatus,
  SequenceGenerateRequest,
  ShotGenerateRequest,
  TextToAnimeRequest,
} from './types';

const BASE_URL = import.meta.env.VITE_API_URL ?? '/api';

export class ApiError extends Error {
  constructor(public status: number, public detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* fall through */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ----- Health -----

export interface HealthInfo {
  name: string;
  version: string;
  warehouse: string;
}

export const getHealth = () => request<HealthInfo>('/');

// ----- Characters -----

export const listCharacters = () => request<CharacterListResponse>('/character/list');

export const getCharacter = (id: string) => request<CharacterResponse>(`/character/${id}`);

export const createCharacter = (body: CharacterCreateRequest) =>
  request<CharacterResponse>('/character/create', {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const deleteCharacter = (id: string) =>
  request<{ status: string; id: string }>(`/character/${id}`, { method: 'DELETE' });

// ----- Generation -----

export const generateShot = (body: ShotGenerateRequest) =>
  request<JobStatus>('/generate/shot', { method: 'POST', body: JSON.stringify(body) });

export const generateSequence = (body: SequenceGenerateRequest) =>
  request<JobStatus>('/generate/sequence', { method: 'POST', body: JSON.stringify(body) });

export const generateAnime = (body: TextToAnimeRequest) =>
  request<JobStatus>('/generate/anime', { method: 'POST', body: JSON.stringify(body) });

// ----- Jobs -----

export const getJob = (id: string) => request<JobStatus>(`/job/${id}`);
