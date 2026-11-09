/* Small enhancements shared by the docs pages: copy buttons on code blocks, tabs, the
   reference page's sidebar highlight, and inlining the IAM policy from its JSON file. */

(function () {
  "use strict";

  // A "Copy" button on every code block.
  document.querySelectorAll("pre").forEach((pre) => {
    const button = document.createElement("button");
    button.className = "copy";
    button.type = "button";
    button.textContent = "Copy";
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector("code").innerText);
        button.textContent = "Copied";
      } catch (e) {
        button.textContent = "Select & copy";
      }
      setTimeout(() => (button.textContent = "Copy"), 1500);
    });
    pre.appendChild(button);
  });

  // Tabs: one panel visible at a time, arrow keys move between tabs.
  document.querySelectorAll("[data-tabs]").forEach((box) => {
    const tabs = [...box.querySelectorAll('[role="tab"]')];
    const select = (tab) => {
      tabs.forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
    };
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (e) => {
        const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!step) return;
        const next = tabs[(i + step + tabs.length) % tabs.length];
        select(next);
        next.focus();
      });
    });
  });

  // Reference page: inline the IAM policy so the JSON file stays the single source of truth.
  const policy = document.getElementById("policy");
  if (policy) {
    fetch("notebook-role-policy.json")
      .then((r) => r.text())
      .then((text) => (policy.textContent = text))
      .catch(() => (policy.textContent = "Open notebook-role-policy.json next to this page."));
  }

  // Reference page: highlight the section you're reading in the sidebar.
  const links = [...document.querySelectorAll("aside.toc a")];
  if (links.length) {
    const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id));
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    targets.forEach((t) => observer.observe(t));
  }
})();
