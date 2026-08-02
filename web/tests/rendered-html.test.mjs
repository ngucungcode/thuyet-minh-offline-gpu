import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

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
  assert.match(html, /Biến một bản phim thành bản thuyết minh Việt/);
  assert.match(html, /Chỉ nội dung bạn có quyền sử dụng/);
  assert.match(html, /Tự động nhận diện/);
  assert.match(html, /Tôi xác nhận mình sở hữu hoặc được phép tải/);
  assert.match(html, /Kho model cục bộ/);
  assert.match(html, /Indexer và phụ đề/);
  assert.match(html, /OpenSubtitles/);
  assert.match(html, /<meta property="og:image" content="http:\/\/localhost(?::3000)?\/og\.png"/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Your site is taking shape/i);
});

test("keeps the dashboard API contract local-first", async () => {
  const [page, layout, route, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/v1/[...path]/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /api<Health>\("\/health"\)/);
  assert.match(page, /api<\{ items: Job\[\] \}>\("\/jobs\?limit=20/);
  assert.match(page, /rights_confirmed: true/);
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
