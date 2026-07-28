# 📘 **README_DEV.md**  

## *Prompt Helper + Fooocus Integration Developer Guide*  

### *(Drop this file directly into your repo)*

---

## Prompt Helper + Fooocus Integration  

### Developer Architecture Guide  

### *(Stable, Remote‑Safe, and Easy to Maintain)*

This document explains how the Prompt Helper standalone app integrates with Fooocus, how the URL structure works, and how to avoid the common pitfalls that caused remote rendering failures.

It is written for future maintainers — including future‑you — so you never have to rediscover this logic again.

---

## 🧩 1. System Overview

The system consists of:

1. **Fooocus Gradio UI**  
2. **Prompt Helper Frontend** (`prompt_helper/static/index.html`)  
3. **Prompt Helper Backend API** (`prompt_helper/app.py`)  
4. **A1111 Extension JS** (`sd-webui-prompt-all-in-one/javascript/main.entry.js`)  

All components run inside a single FastAPI server.

---

## 🧭 2. Required URL Structure

The Prompt Helper frontend and backend expect **exactly** these URLs:

### Frontend Part

```t
/prompt-helper/                 → index.html
/prompt-helper/static/css/*     → CSS files
/prompt-helper/static/js/*      → JS files
```

### Backend API

```t
/prompt-helper/physton_prompt/get_config
/prompt-helper/physton_prompt/get_favorites
/prompt-helper/physton_prompt/styles?file=...
```

### Extension JS

```t
/sd-webui-prompt-all-in-one-js
```

### Fooocus UI

```t
/
```

This structure is fixed and must not be changed unless you also update the frontend code.

---

## 🧱 3. Why Local Worked but Remote Failed

Originally, `index.html` used **relative paths**:

```html
./css/main.min.css
./js/main.js
```

Locally, inside a Gradio iframe, these sometimes resolved to:

```t
/prompt-helper/css/main.min.css
```

Remote browsers resolve strictly:

```t
/prompt-helper/css/main.min.css → 404
```

Because the server actually served:

```t
/prompt-helper/static/css/main.min.css
```

This caused:

- CSS not loading  
- JS not loading  
- Vue/React app not initializing  
- Tabs disappearing  
- Backend calls failing  

The backend was correct.  
The server was correct.  
The mounts were correct.  
**The frontend paths were wrong.**

---

## 🛠️ 4. Required Fix in index.html

Use **absolute paths**, not relative ones:

### Correct

```html
<link rel="stylesheet" href="/prompt-helper/static/css/main.min.css">
<script src="/prompt-helper/static/js/main.js"></script>
```

This makes remote and local behave identically.

---

## 🧩 5. launch.py Architecture (version 0.15)

The server must mount components in this order:

### 1. Static files first  

So backend does not shadow them.

### 2. index.html at `/prompt-helper/`

### 3. Extension JS at `/sd-webui-prompt-all-in-one-js`

### 4. Backend at `/prompt-helper`  

Backend internally defines `/physton_prompt/*`.

### 5. Fooocus UI at `/`

This order is essential.

---

## 🧼 6. What Can Be Removed

These parts are unnecessary:

- `sys.path.insert(0, aio_root)`  
- duplicate `launch.py` inside `prompt_helper/`  
- misleading print lines  
- any unused files inside `prompt_helper/`

Everything else is required.

---

## 🔒 7. Robustness Checklist

### Frontend

- Always use absolute paths (`/prompt-helper/static/...`)
- Never use relative paths (`./css/...`)
- Keep API base paths unchanged

### Backend

- Always mount backend at `/prompt-helper`
- Never mount backend at `/prompt-helper-api`, `/prompt-helper/ui`, etc.

### Static

- Always mount static BEFORE backend
- Always mount static at `/prompt-helper/static`

### Extension

- Keep extension JS at `/sd-webui-prompt-all-in-one-js`

### Testing

- Test both local and remote after any change
- Check browser console for missing CSS/JS
- Check server logs for 404s

---

## 🧭 8. Architecture Diagram

```t
FastAPI Server
│
├── /                      → Fooocus Gradio UI
│
├── /sd-webui-prompt-all-in-one-js
│       → sd-webui-prompt-all-in-one/javascript/main.entry.js
│
└── /prompt-helper
        │
        ├── index.html
        │
        ├── /static
        │      ├── css/*
        │      └── js/*
        │
        └── /physton_prompt
               ├── get_config
               ├── get_favorites
               └── styles?file=...
```

---

## 🧩 9. Summary

- Your original backend mount was correct.  
- The remote failure was caused by index.html, not launch.py.  
- Absolute static paths fix remote rendering.  
- launch.py version 0.15 is stable and minimal.  
- This document ensures future‑you never struggles with this again.

---
