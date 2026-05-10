/**
 * TypeScript mirror of api/schemas/models.py
 * Keep in sync when the Pydantic models change.
 */

// ------- Characters -------

export interface CharacterCreateRequest {
  name: string;
  description?: string;
  image_paths: string[];
}

export interface CharacterResponse {
  id: string;
  name: string;
  shot_count: number;
  has_lora: boolean;
  created: string;
  last_used: string;
}

export interface CharacterListResponse {
  characters: CharacterResponse[];
  total: number;
}

// ------- Shot / Sequence / Text-to-Anime -------

export interface ShotGenerateRequest {
  description: string;
  characters?: string[];
  pose_ref?: string | null;
  width?: number;
  height?: number;
  num_frames?: number;
}

export interface ShotResult {
  video_path?: string | null;
  shot_index: number;
  prompt: string;
  quality_score: number;
  status: string;
}

export interface SequenceGenerateRequest {
  script: string;
  story_id?: string | null;
}

export interface SequenceResult {
  story_id: string;
  shots: ShotResult[];
  final_video?: string | null;
  character_count: number;
}

export type Quality = 'draft' | 'standard' | 'high';
export type Resolution = '480p' | '720p' | '1080p';

export interface TextToAnimeRequest {
  text: string;
  quality?: Quality;
  target_resolution?: Resolution;
  target_fps?: number;
  story_id?: string | null;
}

export interface TextToAnimeResult {
  story_id: string;
  shots: ShotResult[];
  final_video?: string | null;
  character_count: number;
  generated_script?: string | null;
}

// ------- Jobs -------

export type JobStatusName = 'pending' | 'running' | 'completed' | 'failed';

export interface JobStatus {
  job_id: string;
  status: JobStatusName;
  progress: number;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export function isTerminalStatus(s: JobStatusName): boolean {
  return s === 'completed' || s === 'failed';
}
