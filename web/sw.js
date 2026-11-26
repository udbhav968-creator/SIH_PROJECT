const CACHE="road-shield-v1";
const SHELL=["/web/app.css","/web/app.js","/web/config.js","/manifest.webmanifest","/icon.svg"];
self.addEventListener("install",e=>{e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL).catch(()=>{})));self.skipWaiting();});
self.addEventListener("activate",e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==CACHE).map(x=>caches.delete(x)))));self.clients.claim();});
self.addEventListener("fetch",e=>{const u=new URL(e.request.url);if(e.request.method!=="GET"||u.pathname.startsWith("/api/"))return;e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)).catch(()=>fetch(e.request)));});
