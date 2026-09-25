// Prompt Helper runs in an iframe that takes its light/dark look from a `__theme` query parameter.
// The parameter itself is resolved before Gradio boots by a script that launch.py injects into
// <head> (see docs/README_DEV.md, "Theme") and published as window.__refocus_theme_resolved; this
// file only has to hand that value to the iframe. Gradio mounts the iframe when its tab first
// renders, so watch for the element instead of assuming it exists.
(function () {
    function theme() {
        if (window.__refocus_theme_resolved) return window.__refocus_theme_resolved;
        var explicit = new URLSearchParams(window.location.search).get('__theme');
        if (explicit === 'dark' || explicit === 'light') return explicit;
        return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
    }

    function applyTheme() {
        var frame = document.getElementById('prompt-helper-iframe');
        var value = theme();
        if (!frame || frame.dataset.refocusTheme === value) return;
        frame.dataset.refocusTheme = value;
        frame.src = '/prompt-helper/?__theme=' + value;
    }

    function watch() {
        applyTheme();
        new MutationObserver(applyTheme).observe(document.body, {childList: true, subtree: true});
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
                if (!window.__refocus_theme) {   // nothing pinned: follow the system
                    window.__refocus_theme_resolved = null;
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
