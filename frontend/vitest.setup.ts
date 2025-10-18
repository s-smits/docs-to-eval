import "@testing-library/jest-dom";

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (typeof window !== "undefined" && !("ResizeObserver" in window)) {
  (window as any).ResizeObserver = ResizeObserver;
}

if (!("ResizeObserver" in globalThis)) {
  (globalThis as any).ResizeObserver = ResizeObserver;
}
