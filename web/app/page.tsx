"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type Health = {
  status: "ok" | "degraded" | "error";
  model_catalog?: { count?: number; status?: string };
  acquisition_configured?: boolean;
  gpu?: {
    name?: string;
    gpu_name?: string;
    memory_total_mib?: number;
    vram_mib?: number;
    ready?: boolean;
  };
};

type Release = {
  release_id: string;
  title: string;
  size_bytes?: number | null;
  seeders?: number | null;
  published_at?: string | null;
};

type Job = {
  id: string;
  release_id: string;
  status: string;
  stage: string;
  progress_permille: number;
  spec?: { search_query?: string; source_language?: string };
  details?: Record<string, unknown>;
  error?: { message?: string } | null;
  result?: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

const stageOrder = [
  "acquisition",
  "transcription",
  "translation",
  "separation",
  "narration",
  "export",
];

const stageLabels: Record<string, string> = {
  acquisition: "Lấy nguồn",
  transcription: "Nhận dạng",
  translation: "Dịch lời",
  separation: "Tách thoại",
  narration: "Tạo giọng",
  export: "Dựng MP4",
  completed: "Hoàn tất",
};

const statusLabels: Record<string, string> = {
  queued: "Đang chờ",
  downloading: "Đang tải",
  running: "Đang xử lý",
  paused: "Tạm dừng",
  failed: "Có lỗi",
  cancelled: "Đã hủy",
  completed: "Hoàn tất",
  needs_language: "Cần chọn ngôn ngữ",
  needs_subtitle_selection: "Cần chọn phụ đề",
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/v1${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      ...(init?.body ? { "content-type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.detail?.message ?? payload?.message;
    throw new Error(message || `Yêu cầu thất bại (${response.status})`);
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

function shortId(value: string) {
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-5)}` : value;
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [query, setQuery] = useState("");
  const [year, setYear] = useState("");
  const [results, setResults] = useState<Release[]>([]);
  const [selected, setSelected] = useState<Release | null>(null);
  const [releaseId, setReleaseId] = useState("");
  const [sourceMode, setSourceMode] = useState<"search" | "release">("search");
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [subtitleMode, setSubtitleMode] = useState("prefer");
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [searching, setSearching] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const refreshOverview = useCallback(async () => {
    const [healthResult, jobsResult] = await Promise.allSettled([
      api<Health>("/health"),
      api<{ items: Job[] }>("/jobs?limit=12&newest_first=true"),
    ]);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else setHealth(null);
    if (jobsResult.status === "fulfilled") setJobs(jobsResult.value.items);
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refreshOverview(), 0);
    const timer = window.setInterval(() => void refreshOverview(), 3000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refreshOverview]);

  const activeJob = useMemo(
    () => jobs.find((job) => !["completed", "failed", "cancelled"].includes(job.status)),
    [jobs],
  );

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
      setSelected(null);
      if (!payload.results.length) setNotice("Không tìm thấy nguồn phù hợp.");
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "Không thể tìm nguồn");
    } finally {
      setSearching(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const effectiveReleaseId =
      sourceMode === "search" ? selected?.release_id : releaseId.trim();
    if (!effectiveReleaseId) {
      setProblem("Hãy chọn một kết quả hoặc nhập Release ID.");
      return;
    }
    if (!rightsConfirmed) {
      setProblem("Bạn cần xác nhận có quyền tải và xử lý nội dung.");
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
          models: {},
          voice: null,
          voice_rights_confirmed: false,
        }),
      });
      setNotice(`Đã tạo job ${shortId(job.id)}. Tiến trình sẽ tự cập nhật.`);
      await refreshOverview();
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "Không thể tạo job");
    } finally {
      setSubmitting(false);
    }
  }

  async function jobAction(job: Job, action: "cancel" | "resume") {
    setProblem(null);
    try {
      await api<Job>(`/jobs/${job.id}/${action}`, { method: "POST", body: "{}" });
      await refreshOverview();
    } catch (error) {
      setProblem(error instanceof Error ? error.message : "Không thể cập nhật job");
    }
  }

  const gpuName = health?.gpu?.name ?? health?.gpu?.gpu_name ?? "GPU chưa kết nối";
  const vram = health?.gpu?.memory_total_mib ?? health?.gpu?.vram_mib;
  const online = health?.status === "ok" || health?.status === "degraded";

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
            Tách lời diễn viên, giữ nhạc và hiệu ứng, dịch cục bộ rồi dựng lại
            một track âm thanh hoàn chỉnh — không gửi nội dung lên dịch vụ AI.
          </p>
        </div>
        <div className="machine-card" aria-label="Trạng thái máy xử lý">
          <span className="machine-label">MÁY XỬ LÝ</span>
          <strong>{gpuName}</strong>
          <div className="machine-meta">
            <span>{vram ? `${Math.round(vram / 1024)} GB VRAM` : "VRAM đang kiểm tra"}</span>
            <span>{health?.model_catalog?.count ?? 0} model</span>
          </div>
          <div className="signal" aria-hidden="true">
            {Array.from({ length: 18 }).map((_, index) => (
              <i key={index} style={{ height: `${18 + ((index * 17) % 34)}%` }} />
            ))}
          </div>
        </div>
      </section>

      <section className="workspace">
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
              Tìm trong nguồn đã cấu hình
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
                      className={`release-row ${selected?.release_id === release.release_id ? "selected" : ""}`}
                      onClick={() => setSelected(release)}
                    >
                      <span className="release-check" aria-hidden="true">
                        {selected?.release_id === release.release_id ? "✓" : ""}
                      </span>
                      <span className="release-title">
                        <strong>{release.title}</strong>
                        <small>{formatBytes(release.size_bytes)} · {release.seeders ?? 0} seed</small>
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
              <small>Magnet và tệp .torrent trực tiếp cần endpoint nhập nguồn của API.</small>
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
                  <option value="auto">Tự động nhận diện</option>
                  <option value="en">Tiếng Anh</option>
                  <option value="ja">Tiếng Nhật</option>
                  <option value="ko">Tiếng Hàn</option>
                  <option value="th">Tiếng Thái</option>
                  <option value="ar">Tiếng Ả Rập</option>
                  <option value="vi">Tiếng Việt</option>
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

        <aside className="queue panel">
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
                const activeStage = stageOrder.indexOf(job.stage);
                const title = job.spec?.search_query || shortId(job.release_id);
                return (
                  <article className={`job-card status-${job.status}`} key={job.id}>
                    <div className="job-topline">
                      <span className="job-status">{statusLabels[job.status] ?? job.status}</span>
                      <span className="job-percent">{percent.toFixed(0)}%</span>
                    </div>
                    <h3>{title}</h3>
                    <p className="job-id">JOB {shortId(job.id)}</p>
                    <div className="progress-track" aria-label={`Tiến trình ${percent.toFixed(0)} phần trăm`}>
                      <i style={{ width: `${percent}%` }} />
                    </div>
                    <div className="stage-track" aria-label="Các công đoạn xử lý">
                      {stageOrder.map((stage, index) => (
                        <span
                          key={stage}
                          className={index < activeStage ? "done" : index === activeStage ? "current" : ""}
                          title={stageLabels[stage]}
                        >
                          <i />
                          <small>{stageLabels[stage]}</small>
                        </span>
                      ))}
                    </div>
                    {job.error?.message && <p className="job-error">{job.error.message}</p>}
                    <div className="job-actions">
                      {job.status === "completed" && (
                        <a className="small-button download" href={`/v1/jobs/${job.id}/artifacts/video`}>
                          Tải MP4
                        </a>
                      )}
                      {["paused", "failed"].includes(job.status) && (
                        <button className="small-button" type="button" onClick={() => void jobAction(job, "resume")}>
                          Tiếp tục
                        </button>
                      )}
                      {!["completed", "cancelled"].includes(job.status) && (
                        <button className="small-button danger" type="button" onClick={() => void jobAction(job, "cancel")}>
                          Hủy
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

      <footer>
        <span>LỒNG TIẾNG GPU · LOCAL-FIRST</span>
        <span>Dữ liệu media ở lại trên máy xử lý</span>
      </footer>
    </main>
  );
}
