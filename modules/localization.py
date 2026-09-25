import json
import os


current_translation = {}
localization_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'language')


def _load(path):
    """Reads a translation dictionary, returning {} and complaining when it is broken."""
    try:
        with open(path, encoding='utf-8') as f:
            loaded = json.load(f)
    except Exception as e:
        print(f'[i18n] Failed to load {path}: {e}')
        return {}

    if not isinstance(loaded, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                              for k, v in loaded.items()):
        print(f'[i18n] {path} must be a flat JSON object of string -> string; ignored.')
        return {}
    return loaded


def localization_js(filename):
    """Emits `window.localization = {...}` for the head of the page.

    `filename` is the stem of a file in `language/`: `--language zh` reads `language/zh.json`.
    The stem `default` is the user's own override and may be absent (`language/default.json` is
    gitignored), so it is silent; any other missing file is reported, because a typo there would
    otherwise look like "the translation quietly does nothing". `javascript/localization.js` is
    what consumes the emitted dictionary.
    """
    global current_translation

    if isinstance(filename, str) and filename:
        name = os.path.abspath(os.path.join(localization_root, filename + '.json'))
        if os.path.exists(name):
            current_translation = _load(name)
            if current_translation:
                print(f'[i18n] Loaded {len(current_translation)} translations from {name}.')
        elif filename != 'default':
            print(f'[i18n] No language file at {name}; the UI stays in English.')

    return f"window.localization = {json.dumps(current_translation)}"
