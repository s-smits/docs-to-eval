import { describe, it, beforeEach, afterEach, vi, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { act } from "react";
import { DocsToEvalApp } from "../docs-to-eval-app";

describe("DocsToEvalApp", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => [],
      })),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders the primary navigation tabs", async () => {
    await act(async () => {
      render(<DocsToEvalApp />);
    });

    expect(await screen.findByText(/docs-to-eval/i)).toBeInTheDocument();
    expect(
      await screen.findByRole("tab", { name: /Evaluate Corpus/i }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("tab", { name: /Classify Corpus/i }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("tab", { name: /Evaluation Runs/i }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("tab", { name: /Settings/i }),
    ).toBeInTheDocument();
  });
});
