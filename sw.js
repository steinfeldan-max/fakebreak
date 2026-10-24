// FakeBreak service worker: lets the installed app open even on a weak connection, and receives
// files shared to it from other apps (Android "Share" → FakeBreak).
// Network first (so updates arrive right away), falling back to the last copy we saw.
// Model files are left alone: the app caches those itself.
const SHELL = 'fakebreak-shell-v2';
const SHARE = 'fakebreak-share';

self.addEventListener('install', event => {
  event.waitUntil(caches.open(SHELL).then(cache => cache.addAll(['./', 'pella.js', 'icon-192.png'])));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});

// Share target: keep the shared file (or text) for the page, then open the app on it.
async function receiveShare(request) {
  const form = await request.formData();
  const cache = await caches.open(SHARE);
  await cache.delete('file');
  await cache.delete('text');
  const file = form.get('media');
  if (file && file.size) {
    await cache.put('file', new Response(file, {
      headers: { 'Content-Type': file.type || 'application/octet-stream', 'X-File-Name': encodeURIComponent(file.name || 'shared') }
    }));
  }
  const text = [form.get('title'), form.get('text'), form.get('url')].filter(Boolean).join('\n');
  if (text) await cache.put('text', new Response(text));
  return Response.redirect('./?shared=1', 303);
}

self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method === 'POST' && url.searchParams.has('share-target')) {
    event.respondWith(receiveShare(event.request));
    return;
  }
  if (event.request.method !== 'GET' || url.origin !== location.origin || url.pathname.includes('/voice-model/')) return;
  event.respondWith(
    fetch(event.request)
      .then(res => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(SHELL).then(cache => cache.put(event.request, copy));
        }
        return res;
      })
      .catch(() => caches.match(event.request))
  );
});
