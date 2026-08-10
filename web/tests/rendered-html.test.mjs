import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

function assertContainsAll(source, fragments) {
  for (const fragment of fragments) {
    assert.ok(
      source.includes(fragment),
      `Expected source to contain: ${fragment}`,
    );
  }
}

function openingJsxTagContaining(source, marker) {
  const markerIndex = source.indexOf(marker);
  assert.notEqual(markerIndex, -1, `Missing JSX marker: ${marker}`);
  const tagStart = source.lastIndexOf("<input", markerIndex);
  const tagEnd = source.indexOf("/>", markerIndex);
  assert.notEqual(tagStart, -1, `Missing <input> before: ${marker}`);
  assert.notEqual(tagEnd, -1, `Missing closing /> after: ${marker}`);
  return source.slice(tagStart, tagEnd + 2);
}

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Vietnamese GPU dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="vi">/i);
  assert.match(html, /<title>Lồng Tiếng GPU Studio<\/title>/i);
  assert.match(html, /THUYẾT MINH NGOẠI TUYẾN · NVIDIA CUDA/);
  assert.doesNotMatch(html, /THUYẾT MINH NGOẠI TUYẾN · RTX/);
  assert.match(html, /Biến một bản phim thành bản thuyết minh Việt/);
  assert.match(html, /Chỉ nội dung bạn có quyền sử dụng/);
  assert.match(html, /Tự động nhận diện/);
  assert.match(html, /Nhịp lời thuyết minh/);
  assert.match(html, /Tự nhiên/);
  assert.match(html, /Khớp chặt/);
  assert.match(html, /Tôi xác nhận mình sở hữu hoặc được phép tải/);
  assert.match(html, /Kho model cục bộ/);
  assert.match(html, /Indexer và phụ đề/);
  assert.match(html, /OpenSubtitles/);
  assert.match(html, /<meta property="og:image" content="http:\/\/localhost(?::3000)?\/og\.png"/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("keeps the dashboard API contract local-first", async () => {
  const [page, styles, layout, route, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/v1/[...path]/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /api<Health>\("\/health"\)/);
  assert.match(page, /gpus\?: GpuDevice\[\]/);
  assert.match(page, /support_tier\?: GpuSupportTier \| null/);
  assert.match(page, /const gpuDevices = health\?\.gpu\?\.gpus \?\? \[\]/);
  assert.match(page, /const gpuReady = health\?\.gpu\?\.ready === true/);
  assert.match(page, /health\?\.status === "ok" && gpuReady/);
  assert.match(page, /const gpuWarnings = \(health\?\.gpu\?\.warnings \?\? \[\]\)/);
  assert.match(page, /maintenance-limited Volta sm_70/);
  assert.match(page, /experimental CMP 170HX support/);
  assert.match(page, /const gpuSupportTier = health\?\.gpu\?\.support_tier \?\? null/);
  assert.match(page, /"maintenance-limited": "Bảo trì giới hạn"/);
  assert.match(page, /experimental: "Thử nghiệm"/);
  assert.doesNotMatch(page, /gpuWarnings\.some/);
  assert.match(page, /Mức hỗ trợ: \{gpuSupportTier \? gpuSupportTierLabel\[gpuSupportTier\]/);
  assert.match(page, /className="gpu-health-warnings"/);
  assert.match(page, /Máy xử lý có cảnh báo/);
  assert.match(page, /gpuDevices\.map\(\(gpu, index\)/);
  assert.match(page, /formatGpuMemory\(gpu\.memory_total_mib\)/);
  assert.match(page, /!gpuReady \|\|/);
  assert.match(page, /GPU chưa sẵn sàng/);
  assert.doesNotMatch(page, /THUYẾT MINH NGOẠI TUYẾN · RTX/);
  assert.match(styles, /\.health-pill\.degraded \.pulse/);
  assert.match(styles, /\.gpu-device-list/);
  assert.match(styles, /\.gpu-health-warnings/);
  assert.match(page, /api<\{ items: Job\[\] \}>\("\/jobs\?limit=20/);
  assert.match(page, /rights_confirmed: true/);
  assert.match(page, /useState<"natural" \| "strict">\("natural"\)/);
  assert.match(page, /timing_profile: timingProfile/);
  assert.match(page, /details\.phase4_step === "timing_rewrite"/);
  assert.match(page, /Đang xử lý/);
  assert.match(page, /timing_rewrite_attempt/);
  assert.match(page, /Lần rút gọn/);
  assert.match(page, /sourceMode === "upload"/);
  assert.match(page, /accept="\.mp4,\.mkv,video\/mp4,video\/x-matroska"/);
  assert.match(page, /accept="\.srt,application\/x-subrip,text\/plain"/);
  assert.match(page, /api<UploadSession>\("\/uploads"/);
  assertContainsAll(page, [
    "`/v1/uploads/${encodeURIComponent(session.id)}/media`",
    "`/v1/uploads/${encodeURIComponent(session.id)}/subtitle`",
    "`/uploads/${encodeURIComponent(session.id)}/finalize`",
  ]);
  assert.match(page, /method: "DELETE"/);
  assert.match(page, /xhr\.upload\.onprogress/);
  assert.match(page, /uploadRequestRef\.current\?\.abort\(\)/);
  assert.match(page, /Khi dùng SRT thủ công, hãy chọn ngôn ngữ nguồn cụ thể/);
  assert.match(page, /fetch\(`\/v1\$\{path\}`/);
  assert.match(page, /href=\{`\/v1\/jobs\/\$\{job\.id\}\/artifacts\/video`\}/);
  assert.match(page, /"subtitle",\s*"asr",\s*"translation"/);
  assert.match(page, /"mix",\s*"export",\s*"verify",\s*"done"/);
  assert.match(page, /needs_language: "Cần chọn ngôn ngữ"/);
  assert.match(page, /needs_subtitle_selection: "Cần chọn phụ đề"/);
  assert.match(page, /\/subtitles\/use-asr/);
  assert.match(page, /\/artifacts\/subtitle/);
  assert.match(page, /\/artifacts\/timing/);
  assert.match(page, /api<ModelCatalog>\("\/models"\)/);
  assert.match(page, /\/admin\/prowlarr\/indexers/);
  assert.match(page, /\/admin\/prowlarr\/test-all/);
  assert.match(page, /\/admin\/opensubtitles/);
  assert.match(page, /DELETE_OPENSUBTITLES_CREDENTIALS/);
  assert.match(page, /cleanup_pending/);
  assert.match(page, /Dọn cấu hình còn sót/);
  assert.match(page, /"X-Dub-Admin-Request": "1"/);
  assert.match(route, /process\.env\.DUB_API_URL \|\| "http:\/\/127\.0\.0\.1:8080"/);
  assert.match(route, /request\.headers\.get\("range"\)/);
  assert.match(route, /request\.headers\.get\("content-length"\)/);
  assert.match(route, /body = hasBody \? request\.body : null/);
  assert.match(route, /requestInit\.duplex = "half"/);
  assert.doesNotMatch(route, /request\.arrayBuffer\(\)/);
  assert.match(route, /admin_proxy_disabled/);
  assert.doesNotMatch(route, /request\.headers\.get\("x-dub-admin-request"\)/);
  assert.match(route, /export const PUT = proxy/);
  assert.match(route, /export const DELETE = proxy/);
  assert.match(route, /backend_unreachable/);
  assert.match(layout, /Lồng Tiếng GPU Studio/);
  assert.match(layout, /summary_large_image/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);

  await access(new URL("../public/og.png", import.meta.url));
});

test("validates local files and exposes detailed cancellable upload progress", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const mediaInput = openingJsxTagContaining(page, "ref={mediaInputRef}");
  const subtitleInput = openingJsxTagContaining(page, "ref={subtitleInputRef}");

  assert.match(page, /\["\.mp4", "\.mkv"\]\.includes\(fileExtension\(file\.name\)\)/);
  assert.match(page, /fileExtension\(file\.name\) !== "\.srt"/);
  assert.match(page, /Video phải là tệp MP4 hoặc MKV/);
  assert.match(page, /Phụ đề thủ công phải là tệp SRT/);
  assert.match(page, /H\.264\/AVC được giữ nguyên không mã hóa lại/);
  assert.match(page, /HEVC SDR được tự động\s+chuyển mã sang H\.264\/AVC/);
  assert.match(page, /AV1, VP9, VP8 và FFV1 bị từ chối/);
  assert.match(page, /Ảnh bìa nhúng và thumbnail được tự động bỏ qua/);
  assert.match(page, /file\.size <= 0/);
  assert.match(page, /phase: "preparing"/);
  assert.match(page, /"media",\s*0,\s*overallTotal/);
  assert.match(page, /"subtitle",\s*mediaFile\.size,\s*overallTotal/);
  assert.match(page, /overallLoaded: baseLoaded \+ loaded/);
  assert.match(page, /speedBytesPerSecond: loaded \/ elapsedSeconds/);
  assert.match(page, /xhr\.onerror/);
  assert.match(page, /xhr\.onabort/);
  assert.match(page, /session\.media_size_bytes === mediaFile\.size/);
  assert.match(page, /session\.subtitle_size_bytes === subtitleFile\.size/);
  assert.match(page, /const shouldDeleteUpload = cancelled/);
  assert.match(page, /error\.code !== "upload_not_found"/);
  assert.match(page, /uploadRequestFingerprintRef\.current/);
  assert.match(page, /uploadedMediaFileRef\.current === mediaFile/);
  assert.match(page, /Cấu hình bị khóa theo phiên này/);
  assert.match(page, /Tệp đã chọn không khớp phiên đang giữ/);
  assert.match(mediaInput, /disabled=\{submitting \|\| uploadConfigurationLocked\}/);
  assert.match(subtitleInput, /disabled=\{submitting \|\| uploadConfigurationLocked\}/);
  assert.match(page, /Xóa phiên tạm/);
  assert.match(page, /uploadFinalizingRef\.current/);
  assertContainsAll(page, [
    "if (uploadCancelledRef.current)",
    "throw new UploadCancelledError()",
  ]);
  assert.doesNotMatch(
    page,
    /api<UploadSession>\("\/uploads", \{\s*method: "POST",\s*signal:/,
  );
  assert.match(page, /phase: "finalizing"/);
  assert.match(page, /phase: "cancelling"/);
  assert.match(page, /Đã hủy tải tệp và xóa dữ liệu tạm trên máy chủ/);
  assert.match(page, /sourceMode === "upload" \? "Không thể tải tệp và tạo job\."/);
});
