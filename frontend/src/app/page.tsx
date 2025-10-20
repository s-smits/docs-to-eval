"use client";

import { DocsToEvalApp } from "@/components/docs-to-eval-app";
import { DocsToEvalAppRedesigned } from "@/components/docs-to-eval-app-dynamic";
import { useState } from "react";

export default function Home() {
  const [useRedesign, setUseRedesign] = useState(true);

  return (
    <div className="relative">
      {useRedesign ? (
        <DocsToEvalAppRedesigned
          key="redesigned"
          onSwitchToClassic={() => setUseRedesign(false)}
        />
      ) : (
        <DocsToEvalApp
          key="classic"
          onSwitchToRedesign={() => setUseRedesign(true)}
        />
      )}
    </div>
  );
}
