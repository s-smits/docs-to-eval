"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface NavCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  isActive: boolean;
  onClick: () => void;
  color: string;
  delay?: number;
}

export function NavCard({
  icon,
  title,
  description,
  isActive,
  onClick,
  color,
  delay = 0,
}: NavCardProps) {
  const gradientParts = color.split(" ");

  return (
    <motion.button
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      whileHover={{ scale: 1.03, translateY: -3 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={cn(
        "relative group overflow-hidden rounded-2xl p-6 transition-all duration-300",
        "bg-gradient-to-r border text-left w-full",
        "shadow-md hover:shadow-2xl",
        isActive
          ? `${color} text-white border-transparent shadow-2xl`
          : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600"
      )}
    >
      {/* Glowing effect when active */}
      {isActive && (
        <motion.div
          className="absolute inset-0 blur-2xl opacity-30"
          animate={{
            scale: [1, 1.05, 1],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            background: `linear-gradient(135deg, ${gradientParts[0].replace("from-", "")} 0%, ${gradientParts[1].replace("to-", "")} 100%)`,
          }}
        />
      )}

      <div className="relative z-10">
        {/* Icon container */}
        <motion.div
          className={cn(
            "mb-4 w-fit p-3 rounded-xl transition-all duration-300",
            isActive
              ? "bg-white/20 backdrop-blur-sm shadow-lg"
              : "bg-slate-100 dark:bg-slate-700 group-hover:scale-110 group-hover:shadow-md"
          )}
          animate={isActive ? { rotate: [0, 5, -5, 0] } : {}}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className={cn(
            "h-8 w-8 transition-colors",
            isActive ? "text-white" : "text-slate-700 dark:text-slate-200"
          )}>
            {icon}
          </div>
        </motion.div>

        {/* Title */}
        <h3
          className={cn(
            "font-bold text-xl mb-2 transition-colors leading-snug",
            isActive
              ? "text-white"
              : "text-slate-900 dark:text-slate-50"
          )}
        >
          {title}
        </h3>

        {/* Description */}
        <p
          className={cn(
            "text-sm transition-colors line-clamp-2 leading-relaxed",
            isActive
              ? "text-white/95"
              : "text-slate-600 dark:text-slate-400"
          )}
        >
          {description}
        </p>

        {/* Active indicator */}
        {isActive && (
          <motion.div
            className="absolute top-4 right-4 w-3 h-3 bg-white rounded-full"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 500, damping: 15 }}
          />
        )}
      </div>
    </motion.button>
  );
}

