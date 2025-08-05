import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "docs-to-eval - Automated LLM Evaluation System",
  description: "Automated LLM Evaluation System with Math-Verify Integration",
  keywords: ["LLM", "evaluation", "AI", "machine learning", "testing"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} font-sans antialiased min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
