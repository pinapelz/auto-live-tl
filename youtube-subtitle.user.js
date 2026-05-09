// ==UserScript==
// @name         Auto-Live-TL YouTube Client
// @namespace    https://example.com/
// @version      1.1
// @description  Auto Translate Live Subtitles
// @author       pinapelz
// @match        https://www.youtube.com/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==

(function () {
  "use strict";

  const PANEL_ID = "altl-subtitle-panel";
  const SUBTITLE_TEXT_ID = "altl-subtitle-text";
  const FOOTER_TEXT_ID = "altl-footer-text";
  const EVENTS_URL = "http://127.0.0.1:5000/events";
  const PANEL_POSITION_KEY = "altl-subtitle-panel-position-v1";

  let eventSource = null;
  let isConnected = false;
  let lastError = null;
  let reconnectTimer = null;
  const MAX_SUBTITLE_LINES = 3;
  const recentLines = [];

  function loadPanelPosition() {
    try {
      const raw = localStorage.getItem(PANEL_POSITION_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (typeof parsed?.x !== "number" || typeof parsed?.y !== "number") return null;
      return { x: parsed.x, y: parsed.y };
    } catch {
      return null;
    }
  }

  function savePanelPosition(x, y) {
    try {
      localStorage.setItem(PANEL_POSITION_KEY, JSON.stringify({ x, y }));
    } catch {
      // ignore storage failures
    }
  }

  function clampPanelPosition(panel, x, y) {
    const margin = 8;
    const panelWidth = panel.offsetWidth || 600;
    const panelHeight = panel.offsetHeight || 120;

    const maxX = Math.max(margin, window.innerWidth - panelWidth - margin);
    const maxY = Math.max(margin, window.innerHeight - panelHeight - margin);

    const clampedX = Math.min(Math.max(margin, x), maxX);
    const clampedY = Math.min(Math.max(margin, y), maxY);

    return { x: clampedX, y: clampedY };
  }

  function setPanelPosition(panel, x, y, persist = false) {
    const pos = clampPanelPosition(panel, x, y);
    panel.style.left = `${pos.x}px`;
    panel.style.top = `${pos.y}px`;
    if (persist) savePanelPosition(pos.x, pos.y);
  }

  function applyInitialPanelPosition(panel) {
    const saved = loadPanelPosition();
    if (saved) {
      setPanelPosition(panel, saved.x, saved.y, false);
      return;
    }

    // Default position: near the top-center.
    const defaultX = Math.max(12, Math.round((window.innerWidth - panel.offsetWidth) / 2));
    const defaultY = 80;
    setPanelPosition(panel, defaultX, defaultY, false);
  }

  function makePanelDraggable(panel, handle) {
    if (!panel || !handle || panel.dataset.dragReady === "1") return;

    panel.dataset.dragReady = "1";

    let dragging = false;
    let dragOffsetX = 0;
    let dragOffsetY = 0;

    handle.style.cursor = "move";

    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      dragging = true;
      handle.setPointerCapture(event.pointerId);

      const rect = panel.getBoundingClientRect();
      dragOffsetX = event.clientX - rect.left;
      dragOffsetY = event.clientY - rect.top;

      panel.style.transition = "none";
      event.preventDefault();
    });

    handle.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const nextX = event.clientX - dragOffsetX;
      const nextY = event.clientY - dragOffsetY;
      setPanelPosition(panel, nextX, nextY, false);
    });

    function endDrag(event) {
      if (!dragging) return;
      dragging = false;
      try {
        handle.releasePointerCapture(event.pointerId);
      } catch {
        // ignored
      }
      panel.style.transition = "";

      const left = parseFloat(panel.style.left || "0");
      const top = parseFloat(panel.style.top || "0");
      setPanelPosition(panel, left, top, true);
    }

    handle.addEventListener("pointerup", endDrag);
    handle.addEventListener("pointercancel", endDrag);

    window.addEventListener("resize", () => {
      const left = parseFloat(panel.style.left || "0");
      const top = parseFloat(panel.style.top || "0");
      setPanelPosition(panel, left, top, true);
    });
  }

  function ensurePanel() {
    let panel = document.getElementById(PANEL_ID);
    if (!panel) {
      panel = document.createElement("div");
      panel.id = PANEL_ID;

      const subtitleText = document.createElement("div");
      subtitleText.id = SUBTITLE_TEXT_ID;
      subtitleText.textContent = "This is a sample subtitle.";

      const footerText = document.createElement("div");
      footerText.id = FOOTER_TEXT_ID;
      footerText.textContent = "Auto-Live-TL - Machine Translated - no translation should be taken as authoritative or quoted verbatim - (drag me)";
      footerText.style.marginTop = "8px";
      footerText.style.fontSize = "12px";
      footerText.style.opacity = "0.7";
      footerText.style.textAlign = "center";
      footerText.style.userSelect = "none";

      panel.appendChild(subtitleText);
      panel.appendChild(footerText);

      panel.style.position = "fixed";
      panel.style.display = "block";
      panel.style.boxSizing = "border-box";
      panel.style.width = "min(76vw, 960px)";
      panel.style.minWidth = "320px";
      panel.style.maxWidth = "960px";
      panel.style.margin = "0";
      panel.style.padding = "10px 14px";
      panel.style.borderRadius = "10px";
      panel.style.background = "rgba(0,0,0,0.62)";
      panel.style.border = "1px solid rgba(255,255,255,0.18)";
      panel.style.color = "var(--yt-spec-text-primary, #f1f1f1)";
      panel.style.fontSize = "16px";
      panel.style.lineHeight = "1.45";
      panel.style.fontWeight = "500";
      panel.style.textAlign = "center";
      panel.style.whiteSpace = "pre-wrap";
      panel.style.wordBreak = "break-word";
      panel.style.zIndex = "2147483646";
      panel.style.backdropFilter = "blur(3px)";

      document.body.appendChild(panel);
      applyInitialPanelPosition(panel);
      makePanelDraggable(panel, footerText);
    } else if (!panel.isConnected) {
      document.body.appendChild(panel);
    }
    return panel;
  }

  function getSubtitleNode(panel) {
    return panel?.querySelector(`#${SUBTITLE_TEXT_ID}`) || null;
  }

  function setSubtitleText(panel, text) {
    const node = getSubtitleNode(panel);
    if (node) node.textContent = text;
  }

  function renderSubtitleHistory(panel) {
    const node = getSubtitleNode(panel);
    if (!node) return;
    node.textContent = "";
    recentLines.forEach((line, index) => {
      const row = document.createElement("div");
      row.textContent = line;
      const opacity = Math.max(0.35, 1 - index * 0.15);
      row.style.opacity = String(opacity);
      if (index > 0) row.style.filter = "grayscale(1)";
      node.appendChild(row);
    });
  }

  function updateSubtitleHistory(panel, text) {
    if (!text) return;
    if (recentLines.length > 0 && recentLines[0] === text) {
      renderSubtitleHistory(panel);
      return;
    }
    recentLines.unshift(text);
    if (recentLines.length > MAX_SUBTITLE_LINES) {
      recentLines.splice(MAX_SUBTITLE_LINES);
    }
    renderSubtitleHistory(panel);
  }

  function connectEventSource(panel) {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    setSubtitleText(panel, "Connecting to local subtitle server...");
    const source = new EventSource(EVENTS_URL);
    eventSource = source;

    source.addEventListener("subtitle", (event) => {
      try {
        const data = JSON.parse(event.data || "{}");
        updateSubtitleHistory(panel, data.text || "");
      } catch (err) {
        lastError = err;
        console.warn("Auto-Live-TL: failed to parse subtitle event", err);
      }
    });

    source.onopen = () => {
      isConnected = true;
      if (recentLines.length > 0) {
        renderSubtitleHistory(panel);
      } else {
        setSubtitleText(panel, "Connected. Waiting for subtitles...");
      }
    };

    source.onerror = () => {
      isConnected = false;
      setSubtitleText(panel, "Connection lost. Reconnecting...");
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      if (!reconnectTimer) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          connectEventSource(panel);
        }, 2000);
      }
    };
  }

  function startListeningIfNeeded(panel) {
    if (isConnected || eventSource) return;
    if (!location.pathname.startsWith("/watch")) return;
    connectEventSource(panel);
  }

  function injectSubtitlePanel() {
    if (!location.pathname.startsWith("/watch")) return false;

    const panel = ensurePanel();
    startListeningIfNeeded(panel);
    return true;
  }

  injectSubtitlePanel();

  const observer = new MutationObserver(() => {
    injectSubtitlePanel();
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  window.addEventListener("yt-navigate-finish", injectSubtitlePanel);
})();
