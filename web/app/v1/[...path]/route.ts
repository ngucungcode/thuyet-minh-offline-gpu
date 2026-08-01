import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const forwardedResponseHeaders = [
  "accept-ranges",
  "content-disposition",
  "content-length",
  "content-range",
  "content-type",
  "etag",
  "last-modified",
];

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const backend = process.env.DUB_API_URL || "http://127.0.0.1:8080";
  const base = backend.endsWith("/") ? backend : `${backend}/`;
  const destination = new URL(`v1/${path.map(encodeURIComponent).join("/")}`, base);
  destination.search = request.nextUrl.search;

  const requestHeaders = new Headers();
  const contentType = request.headers.get("content-type");
  const range = request.headers.get("range");
  if (contentType) requestHeaders.set("content-type", contentType);
  if (range) requestHeaders.set("range", range);

  const hasBody = !["GET", "HEAD"].includes(request.method);
  try {
    const upstream = await fetch(destination, {
      method: request.method,
      headers: requestHeaders,
      body: hasBody ? await request.arrayBuffer() : undefined,
      cache: "no-store",
      redirect: "manual",
    });
    const responseHeaders = new Headers();
    for (const name of forwardedResponseHeaders) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch {
    return Response.json(
      {
        detail: {
          code: "backend_unreachable",
          message: "Không kết nối được API GPU. Hãy kiểm tra DUB_API_URL và trạng thái stack.",
          retryable: true,
        },
      },
      { status: 503 },
    );
  }
}

export const GET = proxy;
export const HEAD = proxy;
export const POST = proxy;
