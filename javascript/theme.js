// Resolves the UI theme before Gradio boots and writes it into the URL.
//
// Why this exists: Gradio's ImageEditor paints its canvas background once, when the Pixi app is
// created, from the theme it has resolved by then ("dark" -> #27272a, anything else -> #ffffff).
// With only a system preference in play, that resolution happens in an effect that runs after the
// canvas exists, so a dark desktop still got a white canvas, and no CSS can reach the WebGL buffer.
// Putting the resolved theme into `?__theme=` before Gradio starts makes it resolve immediately.
//
// Order of precedence, matching Gradio's own: an explicit `?__theme=` in the URL first (that is what
// Gradio's settings panel writes), then `--theme` from the command line, then the system preference.
// Everything downstream reads `window.__refocus_theme_resolved`, so the Prompt Helper iframe can use
// the very same value.
(function () {
    var SYSTEM_QUERY = '(prefers-color-scheme: dark)';
    var explicit = new URLSearchParams(window.location.search).get('__theme');
    if (explicit === 'dark' || explicit === 'light') {
        window.__refocus_theme_resolved = explicit;
        return;
    }

    function systemTheme() {
        return (window.matchMedia && window.matchMedia(SYSTEM_QUERY).matches) ? 'dark' : 'light';
    }

    var theme = (window.__refocus_theme === 'dark' || window.__refocus_theme === 'light')
        ? window.__refocus_theme
        : systemTheme();
    window.__refocus_theme_resolved = theme;

    try {
        var url = new URL(window.location.href);
        url.searchParams.set('__theme', theme);
        window.history.replaceState(null, '', url);
    } catch (e) {
        // Nothing to do: Gradio resolves the theme itself, the canvas just may not follow it.
    }
})();

// Prompt Helper runs in an iframe that takes its light/dark look from the same `__theme` parameter.
// Gradio mounts the iframe when its tab first renders, so watch for the element instead of assuming
// it exists; re-apply when the OS theme changes and nobody has pinned a theme in the URL.
(function () {
    function applyTheme() {
        var frame = document.getElementById('prompt-helper-iframe');
        var theme = window.__refocus_theme_resolved;
        if (!frame || !theme || frame.dataset.refocusTheme === theme) return;
        frame.dataset.refocusTheme = theme;
        frame.src = '/prompt-helper/?__theme=' + theme;
    }

    function watch() {
        applyTheme();
        new MutationObserver(applyTheme).observe(document.body, {childList: true, subtree: true});
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
                if (new URLSearchParams(window.location.search).get('__theme') === window.__refocus_theme_resolved
                    && window.__refocus_theme !== 'dark' && window.__refocus_theme !== 'light') {
                    window.__refocus_theme_resolved = (window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
                    applyTheme();
                }
            });
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', watch);
    } else {
        watch();
    }
})();
