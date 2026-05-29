/* Tiny JS helpers for the Miles docs site */

document.addEventListener("DOMContentLoaded", () => {
  // Scroll-reveal: add .is-visible to landing cards / strips when they enter the viewport.
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries, obs) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add("is-visible");
            obs.unobserve(e.target);
          }
        }
      },
      { rootMargin: "0px 0px -10% 0px", threshold: 0.05 }
    );
    document
      .querySelectorAll(".miles-card, .miles-update, .miles-quote, .miles-strip")
      .forEach((el) => io.observe(el));
  } else {
    // Fallback: just show everything.
    document
      .querySelectorAll(".miles-card, .miles-update, .miles-quote, .miles-strip")
      .forEach((el) => el.classList.add("is-visible"));
  }

  // Hover micro-interaction on update timeline rows.
  document.querySelectorAll(".miles-update").forEach((el) => {
    el.addEventListener("mouseenter", () => el.classList.add("is-hover"));
    el.addEventListener("mouseleave", () => el.classList.remove("is-hover"));
  });
});
