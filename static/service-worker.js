self.addEventListener('install', event => {
    event.waitUntil(
        caches.open('symptom-checker-cache').then(cache => {
            return cache.addAll([
                '/',
                '/static/styles.css',
                '/static/script.js',
                '/static/icons/icon-192x192.png'
            ]);
        })
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(response => {
            return response || fetch(event.request);
        })
    );
});