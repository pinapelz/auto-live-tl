// ==UserScript==
// @name         Auto-Live-TL YouTube Client
// @namespace    https://example.com/
// @version      1.0
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

  let eventSource = null;
  let isConnected = false;
  let lastError = null;
  let reconnectTimer = null;
  const MAX_SUBTITLE_LINES = 3;
  const recentLines = [];

  function getPrimary() {
    return document.querySelector("#columns #primary");
  }

  function getPlayerNode(primary) {
    return (
      primary?.querySelector("#player") ||
      primary?.querySelector("ytd-player") ||
      primary?.querySelector("#movie_player")?.closest("#player") ||
      primary?.querySelector("#movie_player") ||
      null
    );
  }

  function getTitleAnchor(primary) {
    return (
      primary?.querySelector("ytd-watch-metadata") ||
      primary?.querySelector("#title h1")?.closest("ytd-watch-metadata") ||
      primary?.querySelector("#below ytd-watch-metadata") ||
      primary?.querySelector("#below")?.firstElementChild ||
      null
    );
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
      footerText.textContent = "Machine Translated - no translation should be taken as authoritative or quoted verbatim";
      footerText.style.marginTop = "8px";
      footerText.style.fontSize = "12px";
      footerText.style.opacity = "0.7";
      footerText.style.textAlign = "center";

      panel.appendChild(subtitleText);
      panel.appendChild(footerText);

      panel.style.display = "block";
      panel.style.boxSizing = "border-box";
      panel.style.width = "100%";
      panel.style.margin = "8px 0 12px";
      panel.style.padding = "10px 14px";
      panel.style.borderRadius = "10px";
      panel.style.background = "rgba(255,255,255,0.04)";
      panel.style.border = "1px solid rgba(255,255,255,0.12)";
      panel.style.color = "var(--yt-spec-text-primary, #f1f1f1)";
      panel.style.fontSize = "16px";
      panel.style.lineHeight = "1.45";
      panel.style.fontWeight = "500";
      panel.style.textAlign = "center";
      panel.style.whiteSpace = "pre-wrap";
      panel.style.wordBreak = "break-word";
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

  function safePlaceBefore(targetNode, nodeToPlace) {
    if (!targetNode || !targetNode.parentNode || !targetNode.isConnected) return false;

    const parent = targetNode.parentNode;
    if (nodeToPlace.parentNode !== parent || nodeToPlace.nextSibling !== targetNode) {
      parent.insertBefore(nodeToPlace, targetNode);
    }
    return true;
  }

  function safePlaceAfter(targetNode, nodeToPlace) {
    if (!targetNode || !targetNode.parentNode || !targetNode.isConnected) return false;

    const parent = targetNode.parentNode;
    const desiredNext = targetNode.nextSibling;
    if (nodeToPlace.parentNode !== parent || nodeToPlace.previousSibling !== targetNode) {
      parent.insertBefore(nodeToPlace, desiredNext);
    }
    return true;
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

    const primary = getPrimary();
    if (!primary) return false;

    const panel = ensurePanel();

    startListeningIfNeeded(panel);
    const titleAnchor = getTitleAnchor(primary);
    if (safePlaceBefore(titleAnchor, panel)) return true;
    const playerNode = getPlayerNode(primary);
    if (safePlaceAfter(playerNode, panel)) return true;

    return false;
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
