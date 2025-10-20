"use client";

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { EvaluationTab } from "./evaluation-tab";
import { ClassificationTab } from "./classification-tab";
import { RunsTab } from "./runs-tab";
import { SettingsTab } from "./settings-tab";
import { ThemeToggle } from "./theme-toggle";
import {
  BarChart3,
  Users,
  Settings,
  FlaskConical,
  Menu,
  LayoutGrid,
} from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";

type TabValue = "evaluate" | "classify" | "runs" | "settings";

interface DocsToEvalAppRedesignedProps {
  onSwitchToClassic?: () => void;
}

type CardRectMap = Partial<Record<TabValue, DOMRect>>;

export function DocsToEvalAppRedesigned({
  onSwitchToClassic,
}: DocsToEvalAppRedesignedProps) {
  const [activeTab, setActiveTab] = useState<TabValue>("evaluate");
  const [navLayout, setNavLayout] = useState<"horizontal" | "side">("horizontal");
  const [sideNavCollapsed, setSideNavCollapsed] = useState(false);
  const reduceMotion = useReducedMotion();

  const animationDurationMs = 800;
  const animationDurationSec = animationDurationMs / 1000;
  const easingBezier = "cubic-bezier(0.25, 0.46, 0.45, 0.94)";
  const easingCurve: [number, number, number, number] = [0.25, 0.46, 0.45, 0.94];

  const cardRefs = useRef<Record<TabValue, HTMLButtonElement | null>>({
    evaluate: null,
    classify: null,
    runs: null,
    settings: null,
  });
  const previousRectsRef = useRef<CardRectMap | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const previousContentRectRef = useRef<DOMRect | null>(null);
  const asideRef = useRef<HTMLElement | null>(null);
  const cloneAnimationsRef = useRef<Animation[]>([]);

  const navItems = useMemo(
    () => [
    {
      id: "evaluate" as TabValue,
      icon: <FlaskConical className="h-full w-full" />,
      title: "Evaluate Your Documents",
      description: "Run comprehensive evaluations on your text corpus with AI-powered analysis",
      color: "from-blue-500 to-blue-600",
    },
    {
      id: "classify" as TabValue,
      icon: <Users className="h-full w-full" />,
      title: "Classify Your Content",
      description: "Automatically categorize and organize your corpus data",
      color: "from-violet-500 to-purple-600",
    },
    {
      id: "runs" as TabValue,
      icon: <BarChart3 className="h-full w-full" />,
      title: "View Evaluation Results",
      description: "Track and analyze past evaluation runs and metrics",
      color: "from-emerald-500 to-teal-600",
    },
    {
      id: "settings" as TabValue,
      icon: <Settings className="h-full w-full" />,
      title: "Configure Settings",
      description: "Manage API keys, preferences, and system configuration",
      color: "from-slate-600 to-slate-700",
    },
    ],
    []
  );

  const renderContent = () => {
    switch (activeTab) {
      case "evaluate":
        return <EvaluationTab />;
      case "classify":
        return <ClassificationTab />;
      case "runs":
        return <RunsTab />;
      case "settings":
        return <SettingsTab />;
      default:
        return <EvaluationTab />;
    }
  };

  useLayoutEffect(() => {
    if (reduceMotion) {
      previousRectsRef.current = null;
      previousContentRectRef.current = null;
      return;
    }

    const animationDuration = animationDurationMs;
    const easing = easingBezier;

    // Measure current rects after layout change
    const currentRects: CardRectMap = {};
    navItems.forEach((item) => {
      const el = cardRefs.current[item.id];
      if (el) currentRects[item.id] = el.getBoundingClientRect();
    });

    const contentEl = contentRef.current;
    const currentContentRect = contentEl ? contentEl.getBoundingClientRect() : null;

    const previousRects = previousRectsRef.current;
    const previousContentRect = previousContentRectRef.current;

    // Cancel any leftover clone animations
    cloneAnimationsRef.current.forEach((a) => {
      try {
        a.cancel();
      } catch {}
    });
    cloneAnimationsRef.current = [];

    if (previousRects) {
      // Hide all original cards immediately to prevent flash of final size
      navItems.forEach((item) => {
        const el = cardRefs.current[item.id];
        if (el) {
          el.style.opacity = "0";
          el.style.pointerEvents = "none";
        }
      });
      // Clone-based FLIP for smooth, jump-free animation
      navItems.forEach((item) => {
        const el = cardRefs.current[item.id];
        const prev = previousRects[item.id];
        const next = currentRects[item.id];
        if (!el || !prev || !next) return;

        // Create visual clone positioned at previous location
        const clone = el.cloneNode(true) as HTMLElement;
        const cs = getComputedStyle(el);
        
        // Start from exact previous sizes, clamp end more aggressively for horizontal
        const maxEndWidth = navLayout === "horizontal" ? 220 : 280;
        const startWidth = prev.width;
        const startHeight = prev.height;
        const endWidth = Math.min(next.width, maxEndWidth);
        const endHeight = next.height;
        
        Object.assign(clone.style, {
          position: "fixed",
          left: `${prev.left}px`,
          top: `${prev.top}px`,
          width: `${startWidth}px`,
          height: `${startHeight}px`,
          margin: "0",
          transformOrigin: "top left",
          zIndex: "9999",
          pointerEvents: "none",
          boxSizing: "border-box",
          borderRadius: cs.borderRadius,
          boxShadow: cs.boxShadow,
        });
        document.body.appendChild(clone);

        // Calculate straight path with width/height morph (no scale)
        const dx = next.left - prev.left;
        const dy = next.top - prev.top;

        const anim = clone.animate(
          {
            transform: [
              `translate(0px, 0px)`,
              `translate(${dx}px, ${dy}px)`,
            ],
            width: [`${startWidth}px`, `${endWidth}px`],
            height: [`${startHeight}px`, `${endHeight}px`],
            opacity: [1, 1, 0.8, 0], // Fade out gradually in the last 50%
          },
          { duration: animationDuration, easing, fill: "forwards" }
        );

        cloneAnimationsRef.current.push(anim);

        const finalize = () => {
          clone.remove();
          // Fade in the real card
          el.style.pointerEvents = "";
          const fadeIn = el.animate(
            [{ opacity: 0 }, { opacity: 1 }],
            { duration: 400, easing: "ease-out", fill: "forwards" }
          );
          fadeIn.onfinish = () => {
            el.style.opacity = "";
          };
        };
        anim.onfinish = finalize;
        anim.oncancel = finalize;
      });

      // Fade in the sidebar when switching to side layout (slower fade)
      if (navLayout === "side" && asideRef.current) {
        const sidebarAnim = asideRef.current.animate(
          [{ opacity: 0 }, { opacity: 1 }],
          { duration: animationDuration * 0.8, easing: "ease-out", fill: "both" }
        );
        sidebarAnim.onfinish = () => {
          if (asideRef.current) asideRef.current.style.opacity = "";
        };
      }

      // Content panel slides in from right with slower fade
      if (contentEl && currentContentRect && previousContentRect) {
        const deltaX = currentContentRect.left - previousContentRect.left;
        const startOffset = Math.abs(deltaX) + 40;

        contentEl.getAnimations().forEach((animation) => animation.cancel());

        // Slower fade - 80% of animation duration
        const fadeInDuration = animationDuration * 0.8;

        const animation = contentEl.animate(
          [
            {
              transform: `translateX(${startOffset}px)`,
              opacity: 0,
              offset: 0,
            },
            {
              transform: `translateX(${startOffset * 0.2}px)`,
              opacity: 0.5,
              offset: 0.4,
            },
            {
              transform: `translateX(0px)`,
              opacity: 1,
              offset: 1,
            },
          ],
          {
            duration: animationDuration,
            easing: "ease-out",
            fill: "forwards",
          }
        );

        animation.onfinish = () => {
          requestAnimationFrame(() => {
            animation.commitStyles();
            animation.cancel();
          });
        };
      }
    }

    previousRectsRef.current = currentRects;
    previousContentRectRef.current = currentContentRect;
  }, [navLayout, sideNavCollapsed, reduceMotion, navItems]);

  const renderHorizontalLayout = () => (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        className="relative text-center mb-10"
      >
        <div className="absolute right-0 top-0 flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setNavLayout("side")}
            className="text-xs"
          >
            Use Side Nav
          </Button>
          <ThemeToggle />
        </div>

        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="flex items-center justify-center gap-4 mb-3"
        >
          <motion.div
            className="p-3 bg-gradient-to-br from-blue-500 to-violet-600 rounded-2xl shadow-lg"
            whileHover={{ rotate: 360, scale: 1.1 }}
            transition={{ duration: 0.6 }}
          >
            <BarChart3 className="h-8 w-8 text-white" />
          </motion.div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
            docs-to-eval
          </h1>
        </motion.div>
        <p className="text-base text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Automated LLM Evaluation System with Math-Verify Integration
        </p>
      </motion.div>

      {/* Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        {navItems.map((item) => (
          <motion.button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            ref={(el) => {
              cardRefs.current[item.id] = el;
            }}
            className={cn(
              "relative group overflow-hidden rounded-2xl p-6 transition-all duration-300",
              "bg-gradient-to-r border text-left w-full",
              "shadow-md hover:shadow-2xl",
              activeTab === item.id
                ? `${item.color} text-white border-transparent shadow-2xl`
                : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
            )}
            whileHover={{ scale: 1.03, y: -3 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="relative z-10">
              <motion.div
                className={cn(
                  "mb-4 w-fit p-3 rounded-xl transition-all duration-300",
                  activeTab === item.id
                    ? "bg-white/20 backdrop-blur-sm shadow-lg"
                    : "bg-slate-100 dark:bg-slate-700"
                )}
              >
                <div
                  className={cn(
                    "h-8 w-8 transition-colors",
                    activeTab === item.id
                      ? "text-white"
                      : "text-slate-700 dark:text-slate-200"
                  )}
                >
                  {item.icon}
                </div>
              </motion.div>

              <h3
                className={cn(
                  "font-bold text-xl mb-2 transition-colors leading-snug",
                  activeTab === item.id
                    ? "text-white"
                    : "text-slate-900 dark:text-slate-50"
                )}
              >
                {item.title}
              </h3>

              <p
                className={cn(
                  "text-sm transition-colors line-clamp-2 leading-relaxed",
                  activeTab === item.id
                    ? "text-white/95"
                    : "text-slate-600 dark:text-slate-400"
                )}
              >
                {item.description}
              </p>

              {activeTab === item.id && (
                <motion.div
                  className="absolute top-4 right-4 w-3 h-3 bg-white rounded-full"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 500, damping: 15 }}
                />
              )}
            </div>
          </motion.button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          <Card
            ref={contentRef}
            className="shadow-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900"
          >
            <div className="p-6">{renderContent()}</div>
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  );

  const renderSideLayout = () => (
    <div className="min-h-screen flex">
      <motion.aside
        ref={(el) => {
          asideRef.current = el as unknown as HTMLElement;
        }}
        animate={{
          width: sideNavCollapsed ? 80 : 320,
        }}
        transition={{
          duration: animationDurationSec,
          ease: easingCurve,
        }}
        className={cn(
          "relative bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 shadow-lg",
          "flex flex-col"
        )}
      >
        <div className="p-6 border-b border-gray-200 dark:border-gray-800">
          {!sideNavCollapsed ? (
            <div className="flex items-center gap-3">
              <motion.div
                className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg"
                whileHover={{ rotate: 360 }}
                transition={{ duration: animationDurationSec, ease: easingCurve }}
              >
                <BarChart3 className="h-6 w-6 text-white" />
              </motion.div>
              <div>
                <h2 className="font-bold text-lg bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  docs-to-eval
                </h2>
                <p className="text-xs text-muted-foreground">LLM Evaluation</p>
              </div>
            </div>
          ) : (
            <div className="flex justify-center">
              <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                <BarChart3 className="h-6 w-6 text-white" />
              </div>
            </div>
          )}

          {!sideNavCollapsed ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setNavLayout("horizontal")}
              className="mt-4 w-full text-xs"
            >
              Use Horizontal Layout
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setNavLayout("horizontal")}
              className="mt-4 w-full"
            >
              ↔
            </Button>
          )}
        </div>

        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          {navItems.map((item) => (
            <motion.button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              ref={(el) => {
                cardRefs.current[item.id] = el;
              }}
              className={cn(
                "w-full rounded-xl p-4 transition-all duration-300",
                "hover:scale-105 active:scale-95",
                activeTab === item.id
                  ? `bg-gradient-to-r ${item.color} text-white shadow-lg`
                  : "bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700/80"
              )}
            >
              {!sideNavCollapsed ? (
                <div className="flex items-start gap-3">
                  <div
                    className={cn(
                      "p-2 rounded-lg",
                      activeTab === item.id
                        ? "bg-white/20"
                        : "bg-white dark:bg-slate-700"
                    )}
                  >
                    <div className="h-5 w-5">{item.icon}</div>
                  </div>
                  <div className="flex-1 text-left">
                    <div className="font-semibold text-sm mb-1">{item.title}</div>
                    <div
                      className={cn(
                        "text-xs line-clamp-2",
                        activeTab === item.id
                          ? "text-white/90"
                          : "text-gray-600 dark:text-gray-300"
                      )}
                    >
                      {item.description}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex justify-center">
                  <div className="h-6 w-6">{item.icon}</div>
                </div>
              )}
            </motion.button>
          ))}
        </nav>

        <div className="p-4 border-t border-gray-200 dark:border-gray-800 space-y-2 mt-auto">
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSideNavCollapsed(!sideNavCollapsed)}
              className="flex-1"
            >
              <Menu className="h-4 w-4" />
            </Button>
            {!sideNavCollapsed && (
              <div className="flex-1">
                <ThemeToggle />
              </div>
            )}
          </div>
        </div>
      </motion.aside>

      <main className="flex-1 overflow-auto bg-white dark:bg-slate-950">
        <div className="container mx-auto px-6 py-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <Card
                ref={contentRef}
                className="shadow-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900"
              >
                <div className="p-6">{renderContent()}</div>
              </Card>
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );

  return (
    <div className="min-h-screen bg-white dark:bg-slate-950">
      {onSwitchToClassic && (
        <div className="fixed bottom-8 right-8 z-50">
          <Button
            variant="outline"
            size="sm"
            onClick={onSwitchToClassic}
            className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-md shadow-xl border-2 hover:scale-105 transition-transform"
          >
            <LayoutGrid className="h-3.5 w-3.5 mr-2" />
            Switch to Classic
          </Button>
        </div>
      )}
      {navLayout === "horizontal" ? renderHorizontalLayout() : renderSideLayout()}
    </div>
  );
}

