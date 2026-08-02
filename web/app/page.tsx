"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type JsonObject = Record<string, unknown>;

type Health = {
  status: "ok" | "degraded" | "error";
  api_version?: string;
  database?: { status?: string; journal_mode?: string };
  model_catalog?: { count?: number; status?: string };
  acquisition_configured?: boolean;
  coordinator_configured?: boolean;
  gpu?: {
    name?: string;
    gpu_name?: string;
    memory_total_mib?: number;
    vram_mib?: number;
    ready?: boolean;
  };
};

type Capabilities = {
  offline_inference?: boolean;
  one_active_job_per_gpu?: boolean;
  outputs?: string[];
  cinematic_audio?: {
    dialogue_removed?: boolean;
    music_and_effects_preserved?: boolean;
    narration_ducking?: boolean;
  };
};

type Release = {
  release_id: string;
  title: string;
  indexer_id?: number;
  protocol?: string;
  info_hash?: string | null;
  size_bytes?: number | null;
  seeders?: number | null;
  leechers?: number | null;
  published_at?: string | null;
  categories?: string[];
};

type ModelEntry = {
  id: string;
  stage: string;
  backend: string;
  license: string;
  languages?: string[];
  minimum_vram_mib?: number | null;
  installed?: boolean;
  valid?: boolean;
  selectable?: boolean;
  quality_tier?: string;
  compute_type?: string;
};

type ModelCatalog = {
  schema_version: number;
  models: ModelEntry[];
};

type JobError = {
  code?: string;
  message?: string;
  retryable?: boolean;
};

type Job = {
  id: string;
  release_id: string;
  status: string;
  stage: string;
  progress_permille: number;
  spec?: {
    search_query?: string;
    source_language?: string;
    subtitle_mode?: string;
    models?: Record<string, string | null>;
    voice?: { voice_id?: string | null; reference_path?: string | null } | null;
  };
  details?: JsonObject;
  error?: JobError | null;
  result?: JsonObject | null;
  cancel_requested?: boolean;
  created_at: string;
  updated_at: string;
};

type SubtitleCandidate = {
  subtitle_id: string;
  source?: string;
  language?: string;
  format?: string;
  score?: number;
  high_confidence?: boolean;
  release_name?: string;
  fps?: number | null;
  hearing_impaired?: boolean;
  forced?: boolean;
  matched_by?: string[] | string;
};

type LanguageCandidate = {
  language?: string;
  probability?: number;
  confidence?: number;
};

type IntegrationComponent = {
  configured?: boolean;
  editable?: boolean;
  can_manage?: boolean;
  cleanup_pending?: boolean;
  can_delete?: boolean;
};

type IntegrationsStatus = {
  prowlarr?: IntegrationComponent;
  opensubtitles?: IntegrationComponent;
};

type ProwlarrIndexer = {
  id?: number;
  indexer_id?: number;
  name?: string;
  definition_name?: string | null;
  implementation_name?: string | null;
  protocol?: string;
  privacy?: string | null;
  enable?: boolean;
  enabled?: boolean;
  priority?: number;
  status?: string;
  supports_search?: boolean;
  supports_rss?: boolean;
  disabled_until?: string | null;
  most_recent_failure?: string | null;
};

const stageOrder = [
  "acquisition",
  "subtitle",
  "asr",
  "translation",
  "separation",
  "tts",
  "timing",
  "mix",
  "export",
  "verify",
  "done",
];

const stageLabels: Record<string, string> = {
  acquisition: "Lấy nguồn",
  subtitle: "Phụ đề",
  asr: "Nhận dạng",
  translation: "Dịch lời",
  separation: "Tách thoại",
  tts: "Tạo giọng",
  timing: "Khớp nhịp",
  mix: "Phối âm",
  export: "Dựng MP4",
  verify: "Kiểm tra",
  done: "Hoàn tất",
  mt: "Dịch lời",
  "tts-support": "Hỗ trợ tạo giọng",
};

const statusLabels: Record<string, string> = {
  created: "Đã tạo",
  searching: "Đang tìm nguồn",
  awaiting_release_selection: "Chờ chọn nguồn",
  downloading: "Đang tải",
  subtitle_matching: "Đang khớp phụ đề",
  ready_offline: "Sẵn sàng xử lý offline",
  transcribing: "Đang nhận dạng",
  subtitle_selected: "Đã chọn phụ đề",
  ready_translation: "Sẵn sàng dịch",
  translating: "Đang dịch",
  ready_tts: "Sẵn sàng tạo giọng",
  separating: "Đang tách thoại",
  synthesizing: "Đang tạo giọng",
  timing: "Đang khớp thời lượng",
  mixing: "Đang phối âm",
  muxing: "Đang dựng MP4",
  verifying: "Đang kiểm tra đầu ra",
  completed: "Hoàn tất",
  paused: "Tạm dừng",
  failed: "Có lỗi",
  cancelling: "Đang hủy",
  cancelled: "Đã hủy",
  needs_language: "Cần chọn ngôn ngữ",
  needs_subtitle_selection: "Cần chọn phụ đề",
};

const languageOptions = [
  ["auto", "Tự động nhận diện"],
  ["en", "Tiếng Anh"],
  ["ja", "Tiếng Nhật"],
  ["ko", "Tiếng Hàn"],
  ["th", "Tiếng Thái"],
  ["ar", "Tiếng Ả Rập"],
  ["zh", "Tiếng Trung"],
  ["fr", "Tiếng Pháp"],
  ["de", "Tiếng Đức"],
  ["es", "Tiếng Tây Ban Nha"],
  ["vi", "Tiếng Việt"],
] as const;

const modelStages = [
  ["asr", "Nhận dạng lời nói"],
  ["mt", "Dịch sang tiếng Việt"],
  ["separation", "Tách thoại / nhạc nền"],
  ["tts", "Giọng thuyết minh"],
] as const;

const nonActiveStatuses = new Set(["completed", "failed", "cancelled", "paused"]);

class ApiRequestError extends Error {
  code?: string;
  retryable?: boolean;

  constructor(message: string, code?: string, retryable?: boolean) {
    super(message);
    this.name = "ApiRequestError";
    this.code = code;
    this.retryable = retryable;
  }
}

async function api<T>(path: string, init?: RequestInit, admin = false): Promise<T> {
  const response = await fetch(`/v1${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      ...(init?.body ? { "content-type": "application/json" } : {}),
      ...(admin ? { "X-Dub-Admin-Request": "1" } : {}),
      ...init?.headers,
    },
  });
  const payload = (await response.json().catch(() => null)) as
    | { detail?: string | { code?: string; message?: string; retryable?: boolean }; message?: string }
    | null;
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.message ?? payload?.message;
    throw new ApiRequestError(
      message || `Yêu cầu thất bại (${response.status})`,
      typeof detail === "object" ? detail?.code : undefined,
      typeof detail === "object" ? detail?.retryable : undefined,
    );
  }
  return payload as T;
}

function formatBytes(value?: number | null) {
  if (!value) return "Chưa rõ dung lượng";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
}

function formatDate(value?: string | null) {
  if (!value) return "Không rõ ngày";
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString("vi-VN");
}

function shortId(value: string) {
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-5)}` : value;
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function warningMessages(...values: unknown[]): string[] {
  const messages: string[] = [];
  for (const value of values) {
    for (const item of asArray<unknown>(value)) {
      if (typeof item === "string" && item.trim()) {
        messages.push(item.trim());
      } else if (item && typeof item === "object") {
        const record = item as Record<string, unknown>;
        const message = typeof record.message === "string" ? record.message.trim() : "";
        const code = typeof record.code === "string" ? record.code.trim() : "";
        if (message) messages.push(code ? `${code}: ${message}` : message);
      }
    }
  }
  return [...new Set(messages)];
}

function messageOf(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function stagePosition(job: Job) {
  if (job.status === "completed") return stageOrder.length - 1;
  const index = stageOrder.indexOf(job.stage);
  return index < 0 ? 0 : index;
}

function isSkippedTranscriptStage(job: Job, stage: string) {
  const source = job.details?.transcript_source;
  return (source === "asr" && stage === "subtitle") || (source === "subtitle" && stage === "asr");
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalog>({ schema_version: 1, models: [] });
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [year, setYear] = useState("");
  const [results, setResults] = useState<Release[]>([]);
  const [selectedRelease, setSelectedRelease] = useState<Release | null>(null);
  const [releaseId, setReleaseId] = useState("");
  const [sourceMode, setSourceMode] = useState<"search" | "release">("search");
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [subtitleMode, setSubtitleMode] = useState("prefer");
  const [selectedModels, setSelectedModels] = useState<Record<string, string>>({});
  const [voiceId, setVoiceId] = useState("");
  const [voiceReferencePath, setVoiceReferencePath] = useState("");
  const [voiceRightsConfirmed, setVoiceRightsConfirmed] = useState(false);
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [languageChoice, setLanguageChoice] = useState("en");
  const [searching, setSearching] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const [integrations, setIntegrations] = useState<IntegrationsStatus | null>(null);
  const [indexers, setIndexers] = useState<ProwlarrIndexer[]>([]);
  const [integrationBusy, setIntegrationBusy] = useState<string | null>(null);
  const [integrationNotice, setIntegrationNotice] = useState<string | null>(null);
  const [integrationProblem, setIntegrationProblem] = useState<string | null>(null);
  const [openSubApiKey, setOpenSubApiKey] = useState("");
  const [openSubUsername, setOpenSubUsername] = useState("");
  const [openSubPassword, setOpenSubPassword] = useState("");

  const refreshOverview = useCallback(async () => {
    const [healthResult, jobsResult] = await Promise.allSettled([
      api<Health>("/health"),
      api<{ items: Job[] }>("/jobs?limit=20&newest_first=true"),
    ]);
    setHealth(healthResult.status === "fulfilled" ? healthResult.value : null);
    if (jobsResult.status === "fulfilled") setJobs(jobsResult.value.items);
  }, []);

  const refreshCatalog = useCallback(async () => {
    const [modelResult, capabilityResult] = await Promise.allSettled([
      api<ModelCatalog>("/models"),
      api<Capabilities>("/capabilities"),
    ]);
    if (modelResult.status === "fulfilled") setCatalog(modelResult.value);
    if (capabilityResult.status === "fulfilled") setCapabilities(capabilityResult.value);
  }, []);

  const refreshIntegrations = useCallback(async () => {
    setIntegrationProblem(null);
    const [statusResult, indexerResult] = await Promise.allSettled([
      api<IntegrationsStatus>("/admin/integrations", undefined, true),
      api<ProwlarrIndexer[] | { items?: ProwlarrIndexer[]; indexers?: ProwlarrIndexer[] }>(
        "/admin/prowlarr/indexers",
        undefined,
        true,
      ),
    ]);
    if (statusResult.status === "fulfilled") setIntegrations(statusResult.value);
    if (indexerResult.status === "fulfilled") {
      const payload = indexerResult.value;
      setIndexers(Array.isArray(payload) ? payload : payload.items ?? payload.indexers ?? []);
    } else {
      setIndexers([]);
    }
    if (statusResult.status === "rejected" && indexerResult.status === "rejected") {
      setIntegrationProblem(messageOf(statusResult.reason, "Không thể tải trạng thái tích hợp."));
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      void refreshOverview();
      void refreshCatalog();
      void refreshIntegrations();
    }, 0);
    const timer = window.setInterval(() => void refreshOverview(), 3000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refreshCatalog, refreshIntegrations, refreshOverview]);

  const activeJob = useMemo(
    () => jobs.find((job) => !nonActiveStatuses.has(job.status)),
    [jobs],
  );
  const selectedJob = useMemo(
    () =>
      jobs.find((job) => job.id === selectedJobId) ??
      jobs.find((job) =>
        ["needs_language", "needs_subtitle_selection", "failed"].includes(job.status),
      ) ??
      jobs[0] ??
      null,
    [jobs, selectedJobId],
  );
  const validModelCount = catalog.models.filter((model) => model.installed && model.valid).length;

  function modelsFor(stage: string) {
    return catalog.models.filter((model) => model.stage === stage && model.selectable !== false);
  }

  async function handleSearch(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setProblem(null);
    setNotice(null);
    try {
      const payload = await api<{ results: Release[] }>("/search", {
        method: "POST",
        body: JSON.stringify({
          query: query.trim(),
          year: year ? Number(year) : null,
          media_type: "movie",
        }),
      });
      setResults(payload.results);
      setSelectedRelease(null);
      if (!payload.results.length) setNotice("Không tìm thấy nguồn phù hợp.");
    } catch (error) {
      setProblem(messageOf(error, "Không thể tìm nguồn."));
    } finally {
      setSearching(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const effectiveReleaseId =
      sourceMode === "search" ? selectedRelease?.release_id : releaseId.trim();
    if (!effectiveReleaseId) {
      setProblem("Hãy chọn một kết quả hoặc nhập Release ID.");
      return;
    }
    if (!rightsConfirmed) {
      setProblem("Bạn cần xác nhận có quyền tải và xử lý nội dung.");
      return;
    }
    const hasVoiceSelection = Boolean(voiceId.trim() || voiceReferencePath.trim());
    if (hasVoiceSelection && !voiceRightsConfirmed) {
      setProblem("Bạn cần xác nhận có quyền sử dụng giọng tham chiếu.");
      return;
    }
    setSubmitting(true);
    setProblem(null);
    setNotice(null);
    try {
      const job = await api<Job>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          release_id: effectiveReleaseId,
          rights_confirmed: true,
          source_language: sourceLanguage,
          subtitle_mode: subtitleMode,
          models: {
            asr: selectedModels.asr || null,
            translation: selectedModels.mt || null,
            separation: selectedModels.separation || null,
            tts: selectedModels.tts || null,
          },
          voice: hasVoiceSelection
            ? {
                voice_id: voiceId.trim() || null,
                reference_path: voiceReferencePath.trim() || null,
              }
            : null,
          voice_rights_confirmed: hasVoiceSelection && voiceRightsConfirmed,
        }),
      });
      setSelectedJobId(job.id);
      setNotice(`Đã tạo job ${shortId(job.id)}. Tiến trình sẽ tự cập nhật.`);
      await refreshOverview();
    } catch (error) {
      setProblem(messageOf(error, "Không thể tạo job."));
    } finally {
      setSubmitting(false);
    }
  }

  async function jobMutation(job: Job, action: "cancel" | "resume" | "refresh") {
    const key = `${job.id}:${action}`;
    setBusyAction(key);
    setProblem(null);
    try {
      await api<Job>(`/jobs/${encodeURIComponent(job.id)}/${action}`, { method: "POST" });
      await refreshOverview();
    } catch (error) {
      setProblem(messageOf(error, "Không thể cập nhật job."));
    } finally {
      setBusyAction(null);
    }
  }

  async function selectLanguage(job: Job, language: string) {
    const key = `${job.id}:language`;
    setBusyAction(key);
    setProblem(null);
    try {
      await api<Job>(`/jobs/${encodeURIComponent(job.id)}/language`, {
        method: "POST",
        body: JSON.stringify({ language }),
      });
      setNotice(`Đã chọn ${language} làm ngôn ngữ nguồn.`);
      await refreshOverview();
    } catch (error) {
      setProblem(messageOf(error, "Không thể chọn ngôn ngữ."));
    } finally {
      setBusyAction(null);
    }
  }

  async function selectSubtitle(job: Job, subtitleId?: string) {
    const suffix = subtitleId
      ? `/subtitles/${encodeURIComponent(subtitleId)}`
      : "/subtitles/use-asr";
    const key = `${job.id}:subtitle`;
    setBusyAction(key);
    setProblem(null);
    try {
      await api<Job>(`/jobs/${encodeURIComponent(job.id)}${suffix}`, { method: "POST" });
      setNotice(subtitleId ? "Đã chọn phụ đề cho job." : "Đã chuyển sang nhận dạng lời nói bằng ASR.");
      await refreshOverview();
    } catch (error) {
      setProblem(messageOf(error, "Không thể xác nhận nguồn lời thoại."));
    } finally {
      setBusyAction(null);
    }
  }

  async function testProwlarr() {
    setIntegrationBusy("prowlarr");
    setIntegrationProblem(null);
    setIntegrationNotice(null);
    try {
      const result = await api<{ all_ok?: boolean; failed_count?: number }>(
        "/admin/prowlarr/test-all",
        { method: "POST" },
        true,
      );
      setIntegrationNotice(
        result.all_ok
          ? "Toàn bộ indexer Prowlarr đã vượt qua kiểm tra."
          : `Kiểm tra xong: ${result.failed_count ?? 0} indexer báo lỗi.`,
      );
      await refreshIntegrations();
    } catch (error) {
      setIntegrationProblem(messageOf(error, "Không thể kiểm tra Prowlarr."));
    } finally {
      setIntegrationBusy(null);
    }
  }

  async function saveOpenSubtitles(event: FormEvent) {
    event.preventDefault();
    if (!openSubApiKey.trim() || !openSubUsername.trim() || !openSubPassword) {
      setIntegrationProblem("Hãy nhập API key, tên đăng nhập và mật khẩu OpenSubtitles.");
      return;
    }
    setIntegrationBusy("opensubtitles");
    setIntegrationProblem(null);
    setIntegrationNotice(null);
    try {
      const result = await api<{ message?: string; restart_required?: boolean }>(
        "/admin/opensubtitles",
        {
          method: "PUT",
          body: JSON.stringify({
            api_key: openSubApiKey.trim(),
            username: openSubUsername.trim(),
            password: openSubPassword,
          }),
        },
        true,
      );
      setOpenSubApiKey("");
      setOpenSubPassword("");
      setIntegrationNotice(
        result.message ||
          (result.restart_required
            ? "Đã lưu an toàn. Hãy chạy “dub stack restart” để dịch vụ dùng cấu hình mới."
            : "Đã xác thực và lưu cấu hình OpenSubtitles."),
      );
      await refreshIntegrations();
    } catch (error) {
      setIntegrationProblem(messageOf(error, "Không thể cấu hình OpenSubtitles."));
    } finally {
      setIntegrationBusy(null);
    }
  }

  async function removeOpenSubtitles() {
    if (!window.confirm("Xóa API key, token và tuyến API OpenSubtitles đã lưu trên máy chủ?")) return;
    setIntegrationBusy("opensubtitles-delete");
    setIntegrationProblem(null);
    setIntegrationNotice(null);
    try {
      const result = await api<{ message?: string; restart_required?: boolean }>(
        "/admin/opensubtitles",
        {
          method: "DELETE",
          body: JSON.stringify({ confirm: "DELETE_OPENSUBTITLES_CREDENTIALS" }),
        },
        true,
      );
      setIntegrationNotice(
        result.message || "Đã xóa bí mật OpenSubtitles. Khởi động lại stack để áp dụng.",
      );
      await refreshIntegrations();
    } catch (error) {
      setIntegrationProblem(messageOf(error, "Không thể xóa cấu hình OpenSubtitles."));
    } finally {
      setIntegrationBusy(null);
    }
  }

  const gpuName = health?.gpu?.name ?? health?.gpu?.gpu_name ?? "GPU chưa kết nối";
  const vram = health?.gpu?.memory_total_mib ?? health?.gpu?.vram_mib;
  const online = health?.status === "ok" || health?.status === "degraded";
  const subtitleCandidates = asArray<SubtitleCandidate>(selectedJob?.details?.subtitle_candidates);
  const languageCandidates = asArray<LanguageCandidate>(
    selectedJob?.details?.language_detection_candidates ?? selectedJob?.details?.language_candidates,
  );
  const warnings = warningMessages(
    selectedJob?.details?.warnings,
    selectedJob?.details?.subtitle_warnings,
    selectedJob?.result?.warnings,
  );
  const prowlarrStatus = integrations?.prowlarr;
  const openSubStatus = integrations?.opensubtitles;

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Lồng Tiếng GPU — về đầu trang">
          <span className="brand-mark">LT</span>
          <span>
            <strong>Lồng Tiếng</strong>
            <small>GPU studio</small>
          </span>
        </a>
        <nav className="topnav" aria-label="Điều hướng dashboard">
          <a href="#workflow">Workflow</a>
          <a href="#jobs">Job</a>
          <a href="#models">Model</a>
          <a href="#integrations">Tích hợp</a>
        </nav>
        <div className={`health-pill ${online ? "online" : "offline"}`}>
          <span className="pulse" aria-hidden="true" />
          {online ? "Máy xử lý sẵn sàng" : "Chưa kết nối API"}
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">THUYẾT MINH NGOẠI TUYẾN · RTX</p>
          <h1>Biến một bản phim thành bản thuyết minh Việt.</h1>
          <p className="lead">
            Tách lời diễn viên, giữ nhạc và hiệu ứng, dịch cục bộ rồi dựng lại một track
            thuyết minh hoàn chỉnh — không gửi nội dung lên dịch vụ AI.
          </p>
        </div>
        <div className="machine-card" aria-label="Trạng thái máy xử lý">
          <span className="machine-label">MÁY XỬ LÝ</span>
          <strong>{gpuName}</strong>
          <div className="machine-meta">
            <span>{vram ? `${Math.round(vram / 1024)} GB VRAM` : "VRAM đang kiểm tra"}</span>
            <span>{validModelCount}/{catalog.models.length} model sẵn sàng</span>
          </div>
          <div className="machine-flags">
            <span className={health?.database?.status === "ok" ? "ok" : "warn"}>SQLite {health?.database?.journal_mode ?? "?"}</span>
            <span className={health?.acquisition_configured ? "ok" : "warn"}>Nguồn {health?.acquisition_configured ? "đã nối" : "chưa nối"}</span>
            <span className={capabilities?.offline_inference ? "ok" : "warn"}>AI offline</span>
          </div>
          <div className="signal" aria-hidden="true">
            {Array.from({ length: 18 }).map((_, index) => (
              <i key={index} style={{ height: `${18 + ((index * 17) % 34)}%` }} />
            ))}
          </div>
        </div>
      </section>

      <section className="workspace" id="workflow">
        <div className="composer panel">
          <div className="panel-heading">
            <div>
              <span className="step-number">01</span>
              <h2>Chọn nguồn</h2>
            </div>
            <span className="legal-note">Chỉ nội dung bạn có quyền sử dụng</span>
          </div>

          <div className="source-tabs" role="tablist" aria-label="Cách chọn nguồn">
            <button
              type="button"
              role="tab"
              aria-selected={sourceMode === "search"}
              className={sourceMode === "search" ? "active" : ""}
              onClick={() => setSourceMode("search")}
            >
              Tìm qua Prowlarr
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={sourceMode === "release"}
              className={sourceMode === "release" ? "active" : ""}
              onClick={() => setSourceMode("release")}
            >
              Nhập Release ID
            </button>
          </div>

          {sourceMode === "search" ? (
            <>
              <form className="search-form" onSubmit={handleSearch}>
                <label className="field grow">
                  <span>Tên phim</span>
                  <input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Ví dụ: phim tài liệu của tôi"
                    maxLength={300}
                  />
                </label>
                <label className="field year-field">
                  <span>Năm</span>
                  <input
                    value={year}
                    onChange={(event) => setYear(event.target.value.replace(/\D/g, "").slice(0, 4))}
                    inputMode="numeric"
                    placeholder="2024"
                  />
                </label>
                <button className="primary search-button" disabled={searching || !query.trim()}>
                  {searching ? "Đang tìm…" : "Tìm nguồn"}
                </button>
              </form>

              {results.length > 0 && (
                <div className="results" aria-label="Kết quả tìm nguồn">
                  {results.map((release) => (
                    <button
                      type="button"
                      key={release.release_id}
                      className={`release-row ${selectedRelease?.release_id === release.release_id ? "selected" : ""}`}
                      onClick={() => setSelectedRelease(release)}
                    >
                      <span className="release-check" aria-hidden="true">
                        {selectedRelease?.release_id === release.release_id ? "✓" : ""}
                      </span>
                      <span className="release-title">
                        <strong>{release.title}</strong>
                        <small>
                          {formatBytes(release.size_bytes)} · {release.seeders ?? 0} seed · {release.protocol ?? "nguồn"}
                        </small>
                        <small>
                          Indexer {release.indexer_id ?? "?"} · {formatDate(release.published_at)}
                        </small>
                      </span>
                      <span className="release-id">{shortId(release.release_id)}</span>
                    </button>
                  ))}
                </div>
              )}
            </>
          ) : (
            <label className="field release-input">
              <span>Release ID từ kết quả tìm kiếm trước đó</span>
              <input
                value={releaseId}
                onChange={(event) => setReleaseId(event.target.value)}
                placeholder="Dán Release ID"
                maxLength={500}
              />
              <small>Release ID phải tồn tại trong bộ nhớ nguồn của API.</small>
            </label>
          )}

          <div className="divider" />

          <div className="panel-heading compact">
            <div>
              <span className="step-number">02</span>
              <h2>Cấu hình bản tiếng Việt</h2>
            </div>
          </div>

          <form className="job-form" onSubmit={handleSubmit}>
            <div className="settings-grid">
              <label className="field">
                <span>Ngôn ngữ nguồn</span>
                <select value={sourceLanguage} onChange={(event) => setSourceLanguage(event.target.value)}>
                  {languageOptions.map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Nguồn lời thoại</span>
                <select value={subtitleMode} onChange={(event) => setSubtitleMode(event.target.value)}>
                  <option value="prefer">Ưu tiên phụ đề, fallback ASR</option>
                  <option value="asr">Luôn nhận dạng bằng ASR</option>
                  <option value="manual">Chọn phụ đề thủ công</option>
                </select>
              </label>
            </div>

            <details className="advanced-config">
              <summary>Model và giọng nói nâng cao</summary>
              <div className="model-select-grid">
                {modelStages.map(([stage, label]) => (
                  <label className="field" key={stage}>
                    <span>{label}</span>
                    <select
                      value={selectedModels[stage] ?? ""}
                      onChange={(event) =>
                        setSelectedModels((current) => ({ ...current, [stage]: event.target.value }))
                      }
                    >
                      <option value="">Mặc định của máy chủ</option>
                      {modelsFor(stage).map((model) => (
                        <option key={model.id} value={model.id} disabled={!model.installed || !model.valid}>
                          {model.id}{model.installed && model.valid ? "" : " — chưa sẵn sàng"}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
              <div className="voice-grid">
                <label className="field">
                  <span>Voice ID (tùy chọn)</span>
                  <input value={voiceId} onChange={(event) => setVoiceId(event.target.value)} placeholder="Preset giọng cục bộ" />
                </label>
                <label className="field">
                  <span>Đường dẫn giọng tham chiếu trên server</span>
                  <input
                    value={voiceReferencePath}
                    onChange={(event) => setVoiceReferencePath(event.target.value)}
                    placeholder="/workspace/voices/reference.wav"
                  />
                </label>
              </div>
              {(voiceId || voiceReferencePath) && (
                <label className="rights-check voice-rights">
                  <input
                    type="checkbox"
                    checked={voiceRightsConfirmed}
                    onChange={(event) => setVoiceRightsConfirmed(event.target.checked)}
                  />
                  <span>Tôi có quyền sử dụng giọng hoặc bản ghi tham chiếu này.</span>
                </label>
              )}
              <p className="helper-copy">
                Chỉ model đã cài và xác minh mới chọn được. Cài profile bằng CLI: <code>dub models install-profile maximum --yes</code>.
              </p>
            </details>

            <label className="rights-check">
              <input
                type="checkbox"
                checked={rightsConfirmed}
                onChange={(event) => setRightsConfirmed(event.target.checked)}
              />
              <span>
                Tôi xác nhận mình sở hữu hoặc được phép tải, chỉnh sửa và xử lý nội dung này.
              </span>
            </label>

            {(problem || notice) && (
              <div className={`notice ${problem ? "error" : "success"}`} role="status">
                {problem ?? notice}
              </div>
            )}

            <button
              className="primary start-button"
              disabled={submitting || Boolean(activeJob) || !rightsConfirmed}
            >
              <span>{submitting ? "Đang tạo job…" : activeJob ? "GPU đang bận" : "Bắt đầu thuyết minh"}</span>
              <span aria-hidden="true">→</span>
            </button>
          </form>
        </div>

        <aside className="queue panel" id="jobs">
          <div className="panel-heading queue-heading">
            <div>
              <span className="step-number">03</span>
              <h2>Tiến trình</h2>
            </div>
            <button className="icon-button" type="button" onClick={() => void refreshOverview()} aria-label="Làm mới tiến trình">
              ↻
            </button>
          </div>

          {!jobs.length ? (
            <div className="empty-state">
              <span>00:00</span>
              <strong>Chưa có job nào</strong>
              <p>Job mới sẽ xuất hiện tại đây và tự cập nhật mỗi 3 giây.</p>
            </div>
          ) : (
            <div className="job-list">
              {jobs.map((job) => {
                const percent = Math.max(0, Math.min(100, job.progress_permille / 10));
                const activeStage = stagePosition(job);
                const title = job.spec?.search_query || shortId(job.release_id);
                return (
                  <article
                    className={`job-card status-${job.status} ${selectedJob?.id === job.id ? "selected" : ""}`}
                    key={job.id}
                  >
                    <div className="job-topline">
                      <span className="job-status">{statusLabels[job.status] ?? job.status}</span>
                      <span className="job-percent">{percent.toFixed(0)}%</span>
                    </div>
                    <h3>{title}</h3>
                    <p className="job-id">JOB {shortId(job.id)} · {stageLabels[job.stage] ?? job.stage}</p>
                    <div className="progress-track" aria-label={`Tiến trình ${percent.toFixed(0)} phần trăm`}>
                      <i style={{ width: `${percent}%` }} />
                    </div>
                    <div className="stage-track" aria-label="Các công đoạn xử lý">
                      {stageOrder.map((stage, index) => (
                        <span
                          key={stage}
                          className={
                            isSkippedTranscriptStage(job, stage)
                              ? "skipped"
                              : index < activeStage
                                ? "done"
                                : index === activeStage
                                  ? "current"
                                  : ""
                          }
                          title={stageLabels[stage]}
                        >
                          <i />
                          <small>{stageLabels[stage]}</small>
                        </span>
                      ))}
                    </div>
                    {job.error?.message && <p className="job-error">{job.error.message}</p>}
                    <div className="job-actions">
                      <button className="small-button" type="button" onClick={() => setSelectedJobId(job.id)}>
                        Chi tiết
                      </button>
                      {job.status === "completed" && (
                        <a className="small-button download" href={`/v1/jobs/${job.id}/artifacts/video`}>
                          Tải MP4
                        </a>
                      )}
                      {(job.status === "paused" || (job.status === "failed" && job.error?.retryable)) && (
                        <button
                          className="small-button"
                          type="button"
                          disabled={busyAction === `${job.id}:resume`}
                          onClick={() => void jobMutation(job, "resume")}
                        >
                          Tiếp tục
                        </button>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </aside>
      </section>

      {selectedJob && (
        <section className="panel detail-panel" aria-labelledby="job-detail-title">
          <div className="panel-heading">
            <div>
              <span className="step-number">04</span>
              <h2 id="job-detail-title">Điều khiển job</h2>
            </div>
            <span className={`status-chip status-${selectedJob.status}`}>
              {statusLabels[selectedJob.status] ?? selectedJob.status}
            </span>
          </div>

          <div className="detail-meta-grid">
            <div><span>Job ID</span><strong>{selectedJob.id}</strong></div>
            <div><span>Công đoạn</span><strong>{stageLabels[selectedJob.stage] ?? selectedJob.stage}</strong></div>
            <div><span>Cập nhật</span><strong>{formatDate(selectedJob.updated_at)}</strong></div>
            <div><span>Ngôn ngữ</span><strong>{selectedJob.spec?.source_language ?? "auto"}</strong></div>
          </div>

          <div className="detail-stage-track">
            {stageOrder.map((stage, index) => {
              const activeStage = stagePosition(selectedJob);
              const skipped = isSkippedTranscriptStage(selectedJob, stage);
              return (
                <div
                  key={stage}
                  className={skipped ? "skipped" : index < activeStage ? "done" : index === activeStage ? "current" : ""}
                >
                  <i>{skipped ? "—" : index < activeStage ? "✓" : String(index + 1).padStart(2, "0")}</i>
                  <span>{stageLabels[stage]}</span>
                </div>
              );
            })}
          </div>

          {selectedJob.status === "needs_language" && (
            <div className="action-panel warning-panel">
              <div>
                <span className="action-kicker">CẦN BẠN QUYẾT ĐỊNH</span>
                <h3>Chọn ngôn ngữ chính của lời thoại</h3>
                <p>Whisper chưa đủ chắc chắn. Chọn đúng ngôn ngữ để job tiếp tục mà không chạy lại phần tải nguồn.</p>
              </div>
              {languageCandidates.length > 0 && (
                <div className="candidate-chips">
                  {languageCandidates.map((candidate, index) => {
                    const code = candidate.language ?? "";
                    const confidence = candidate.probability ?? candidate.confidence;
                    return (
                      <button
                        type="button"
                        key={`${code}-${index}`}
                        disabled={!code || busyAction === `${selectedJob.id}:language`}
                        onClick={() => void selectLanguage(selectedJob, code)}
                      >
                        <strong>{code || "?"}</strong>
                        <small>{typeof confidence === "number" ? `${(confidence * 100).toFixed(0)}%` : "đề xuất"}</small>
                      </button>
                    );
                  })}
                </div>
              )}
              <div className="inline-choice">
                <label className="field">
                  <span>Chọn thủ công</span>
                  <select value={languageChoice} onChange={(event) => setLanguageChoice(event.target.value)}>
                    {languageOptions.filter(([value]) => value !== "auto").map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </label>
                <button
                  className="primary"
                  type="button"
                  disabled={busyAction === `${selectedJob.id}:language`}
                  onClick={() => void selectLanguage(selectedJob, languageChoice)}
                >
                  Xác nhận ngôn ngữ
                </button>
              </div>
            </div>
          )}

          {selectedJob.status === "needs_subtitle_selection" && (
            <div className="action-panel warning-panel">
              <div>
                <span className="action-kicker">CẦN BẠN QUYẾT ĐỊNH</span>
                <h3>Chọn phụ đề hoặc dùng ASR</h3>
                <p>Không có ứng viên nào đủ chắc chắn để tự chọn. Hãy kiểm tra tên release, ngôn ngữ và điểm khớp.</p>
              </div>
              <div className="subtitle-candidates">
                {subtitleCandidates.map((candidate) => (
                  <article key={candidate.subtitle_id}>
                    <div>
                      <strong>{candidate.release_name || candidate.subtitle_id}</strong>
                      <p>
                        {candidate.language ?? "?"} · {candidate.format ?? "?"} · nguồn {candidate.source ?? "?"}
                        {typeof candidate.score === "number" ? ` · điểm ${candidate.score.toFixed(2)}` : ""}
                      </p>
                      <small>
                        {candidate.hearing_impaired ? "Có mô tả âm thanh · " : ""}
                        {candidate.forced ? "Forced · " : ""}
                        {Array.isArray(candidate.matched_by) ? candidate.matched_by.join(", ") : candidate.matched_by}
                      </small>
                    </div>
                    <button
                      className="small-button download"
                      type="button"
                      disabled={busyAction === `${selectedJob.id}:subtitle`}
                      onClick={() => void selectSubtitle(selectedJob, candidate.subtitle_id)}
                    >
                      Dùng phụ đề này
                    </button>
                  </article>
                ))}
                {!subtitleCandidates.length && <p className="helper-copy">API chưa trả về ứng viên phụ đề.</p>}
              </div>
              <button
                className="small-button"
                type="button"
                disabled={busyAction === `${selectedJob.id}:subtitle`}
                onClick={() => void selectSubtitle(selectedJob)}
              >
                Bỏ qua phụ đề, dùng ASR
              </button>
            </div>
          )}

          {warnings.length > 0 && (
            <div className="warning-list">
              <strong>Cảnh báo chất lượng</strong>
              <ul>{warnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}</ul>
            </div>
          )}

          {selectedJob.error?.message && (
            <div className="notice error">
              <strong>{selectedJob.error.code ?? "pipeline_error"}</strong>: {selectedJob.error.message}
              {selectedJob.error.retryable === false ? " · Lỗi này không thể thử lại tự động." : ""}
            </div>
          )}

          <div className="detail-actions">
            <button
              className="small-button"
              type="button"
              disabled={busyAction === `${selectedJob.id}:refresh`}
              onClick={() => void jobMutation(selectedJob, "refresh")}
            >
              Làm mới job
            </button>
            {(selectedJob.status === "paused" || (selectedJob.status === "failed" && selectedJob.error?.retryable)) && (
              <button
                className="small-button download"
                type="button"
                disabled={busyAction === `${selectedJob.id}:resume`}
                onClick={() => void jobMutation(selectedJob, "resume")}
              >
                Tiếp tục từ checkpoint
              </button>
            )}
            {!["completed", "cancelling", "cancelled"].includes(selectedJob.status) && (
              <button
                className="small-button danger"
                type="button"
                disabled={busyAction === `${selectedJob.id}:cancel`}
                onClick={() => void jobMutation(selectedJob, "cancel")}
              >
                Hủy job
              </button>
            )}
          </div>

          {selectedJob.status === "completed" && (
            <div className="artifact-panel">
              <div>
                <span className="action-kicker">ĐẦU RA ĐÃ XÁC MINH</span>
                <h3>Tải kết quả</h3>
                <p>MP4 thuyết minh, phụ đề tiếng Việt và báo cáo timing được phục vụ trực tiếp từ máy GPU.</p>
              </div>
              <div className="artifact-actions">
                <a href={`/v1/jobs/${selectedJob.id}/artifacts/video`}>MP4 thuyết minh</a>
                <a href={`/v1/jobs/${selectedJob.id}/artifacts/subtitle`}>Phụ đề SRT</a>
                <a href={`/v1/jobs/${selectedJob.id}/artifacts/timing`}>Timing JSON</a>
              </div>
            </div>
          )}
        </section>
      )}

      <section className="panel inventory-panel" id="models">
        <div className="panel-heading">
          <div>
            <span className="step-number">05</span>
            <h2>Kho model cục bộ</h2>
          </div>
          <button className="small-button" type="button" onClick={() => void refreshCatalog()}>Làm mới danh mục</button>
        </div>
        <p className="section-intro">
          Dashboard chỉ chọn model đã cài và xác minh; việc tải model là thao tác quản trị rõ ràng qua CLI, không xảy ra trong lúc xử lý job.
        </p>
        <div className="model-catalog">
          {catalog.models.map((model) => (
            <article key={model.id} className={model.installed && model.valid ? "ready" : "missing"}>
              <div className="model-title-row">
                <span>{stageLabels[model.stage] ?? model.stage}</span>
                <i>{model.installed && model.valid ? "SẴN SÀNG" : model.installed ? "CHƯA XÁC MINH" : "CHƯA CÀI"}</i>
              </div>
              <h3>{model.id}</h3>
              <p>{model.backend} · {model.compute_type ?? "compute mặc định"}</p>
              <div>
                <small>{model.quality_tier ?? "standard"}</small>
                <small>{model.minimum_vram_mib ? `${Math.ceil(model.minimum_vram_mib / 1024)} GB VRAM` : "CPU"}</small>
                <small>{model.license}</small>
              </div>
            </article>
          ))}
          {!catalog.models.length && <p className="helper-copy">Danh mục model chưa sẵn sàng.</p>}
        </div>
      </section>

      <section className="panel integrations-panel" id="integrations">
        <div className="panel-heading">
          <div>
            <span className="step-number">06</span>
            <h2>Indexer và phụ đề</h2>
          </div>
          <button className="small-button" type="button" onClick={() => void refreshIntegrations()}>Làm mới tích hợp</button>
        </div>
        <p className="section-intro">
          Bí mật được kiểm tra và lưu phía máy chủ; dashboard không đọc lại hoặc hiển thị API key, token hay mật khẩu.
        </p>

        {(integrationProblem || integrationNotice) && (
          <div className={`notice ${integrationProblem ? "error" : "success"}`} role="status">
            {integrationProblem ?? integrationNotice}
          </div>
        )}

        <div className="integration-grid">
          <article className="integration-card">
            <div className="integration-heading">
              <div>
                <span className="action-kicker">INDEXER MANAGER</span>
                <h3>Prowlarr</h3>
              </div>
              <span className={`status-chip ${prowlarrStatus?.configured ? "online" : "offline"}`}>
                {prowlarrStatus?.configured ? "Đã cấu hình" : "Chưa cấu hình"}
              </span>
            </div>
            <p>
              Thêm hoặc sửa indexer trong giao diện Prowlarr. Dashboard chỉ hiển thị cấu hình đã lọc bí mật và gọi kiểm tra toàn bộ.
            </p>
            <ol className="setup-steps">
              <li>Mở Prowlarr qua tunnel localhost.</li>
              <li>Vào <strong>Indexers → Add Indexer</strong>, chọn nhà cung cấp mà bạn được phép sử dụng.</li>
              <li>Nhập URL/thông tin của nhà cung cấp, bấm Test rồi Save.</li>
            </ol>
            <div className="integration-actions">
              <a className="small-button download" href="http://127.0.0.1:9696" target="_blank" rel="noreferrer">
                Mở Prowlarr
              </a>
              <button className="small-button" type="button" disabled={integrationBusy === "prowlarr"} onClick={() => void testProwlarr()}>
                {integrationBusy === "prowlarr" ? "Đang kiểm tra…" : "Test toàn bộ indexer"}
              </button>
            </div>
            <div className="indexer-list">
              {indexers.map((indexer, index) => {
                const enabled = indexer.enabled ?? indexer.enable ?? false;
                const healthy = enabled && !indexer.disabled_until && !indexer.most_recent_failure;
                return (
                  <div key={indexer.id ?? indexer.indexer_id ?? `${indexer.name}-${index}`}>
                    <span className={healthy ? "indexer-dot enabled" : enabled ? "indexer-dot warning" : "indexer-dot"} />
                    <strong title={indexer.most_recent_failure ?? undefined}>
                      {indexer.name ?? `Indexer ${indexer.id ?? indexer.indexer_id ?? index + 1}`}
                    </strong>
                    <small>
                      {indexer.protocol ?? "?"} · ưu tiên {indexer.priority ?? "?"}
                      {indexer.supports_search ? " · search" : ""}
                    </small>
                  </div>
                );
              })}
              {!indexers.length && <p>Chưa có indexer hoặc Prowlarr chưa truy cập được.</p>}
            </div>
            <p className="tunnel-note">
              Máy GPU từ xa: mở tunnel <code>ssh -L 8080:127.0.0.1:8080 -L 9696:127.0.0.1:9696 …</code>, rồi truy cập hai cổng qua localhost.
            </p>
          </article>

          <article className="integration-card">
            <div className="integration-heading">
              <div>
                <span className="action-kicker">SUBTITLE PROVIDER</span>
                <h3>OpenSubtitles</h3>
              </div>
              <span className={`status-chip ${openSubStatus?.configured ? "online" : "offline"}`}>
                {openSubStatus?.cleanup_pending
                  ? "Cần dọn secret cũ"
                  : openSubStatus?.configured
                    ? "Đã cấu hình"
                    : "Chưa cấu hình"}
              </span>
            </div>
            <p>
              Nhập API key và tài khoản để máy chủ đăng nhập, kiểm tra token rồi ghi các file secret với quyền hạn chế. Mật khẩu không được lưu.
            </p>
            {openSubStatus?.cleanup_pending ? (
              <div className="readonly-note">
                {openSubStatus.can_delete === false
                  ? "Cấu hình đã bị vô hiệu hóa nhưng máy chủ hiện không thể dọn file còn sót. Hãy sửa quyền thư mục secret trên host rồi tải lại trang."
                  : "Cấu hình đã bị vô hiệu hóa nhưng còn file dọn dở. Bấm “Dọn cấu hình còn sót” để hoàn tất trước khi đăng nhập lại."}
              </div>
            ) : openSubStatus?.can_manage === false ? (
              <div className="readonly-note">
                Cấu hình đang ở chế độ chỉ đọc (thường do secret được mount từ Docker). Hãy sửa secret ở máy host rồi khởi động lại stack.
              </div>
            ) : null}
            <form className="admin-form" onSubmit={saveOpenSubtitles}>
              <label className="field">
                <span>API key</span>
                <input
                  type="password"
                  autoComplete="off"
                  value={openSubApiKey}
                  onChange={(event) => setOpenSubApiKey(event.target.value)}
                  placeholder={openSubStatus?.configured ? "Đã lưu — nhập key mới để thay" : "OpenSubtitles API key"}
                />
              </label>
              <label className="field">
                <span>Tên đăng nhập</span>
                <input
                  autoComplete="username"
                  value={openSubUsername}
                  onChange={(event) => setOpenSubUsername(event.target.value)}
                  placeholder="Tên tài khoản"
                />
              </label>
              <label className="field">
                <span>Mật khẩu dùng một lần để lấy token</span>
                <input
                  type="password"
                  autoComplete="current-password"
                  value={openSubPassword}
                  onChange={(event) => setOpenSubPassword(event.target.value)}
                  placeholder="Không được lưu trên server"
                />
              </label>
              <div className="integration-actions">
                <button
                  className="primary"
                  disabled={
                    integrationBusy === "opensubtitles" ||
                    openSubStatus?.can_manage === false ||
                    openSubStatus?.cleanup_pending === true
                  }
                >
                  {integrationBusy === "opensubtitles" ? "Đang xác thực…" : "Xác thực và lưu"}
                </button>
                {(openSubStatus?.configured || openSubStatus?.cleanup_pending) &&
                  openSubStatus?.can_delete !== false && (
                  <button
                    className="small-button danger"
                    type="button"
                    disabled={integrationBusy === "opensubtitles-delete"}
                    onClick={() => void removeOpenSubtitles()}
                  >
                    {openSubStatus?.cleanup_pending ? "Dọn cấu hình còn sót" : "Xóa cấu hình"}
                  </button>
                )}
              </div>
            </form>
            <p className="tunnel-note">
              Sau khi đổi cấu hình, chạy <code>dub stack restart</code> nếu API báo cần khởi động lại. Token hết hạn có thể được cấp lại bằng form này.
            </p>
          </article>
        </div>
      </section>

      <footer>
        <span>LỒNG TIẾNG GPU · LOCAL-FIRST</span>
        <span>Dữ liệu media và bí mật ở lại trên máy xử lý</span>
      </footer>
    </main>
  );
}
