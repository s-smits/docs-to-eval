"use client";

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { EvaluationTab } from "./evaluation-tab";
import { ClassificationTab } from "./classification-tab";
import { RunsTab } from "./runs-tab";
import { SettingsTab } from "./settings-tab";
import { ThemeToggle } from "./theme-toggle";
import { BarChart3, Users, Settings, FlaskConical, Sparkles } from "lucide-react";

interface DocsToEvalAppProps {
  onSwitchToRedesign?: () => void;
}

export function DocsToEvalApp({ onSwitchToRedesign }: DocsToEvalAppProps) {
  return (
    <div className="min-h-screen bg-white dark:bg-slate-950">
      {onSwitchToRedesign && (
        <div className="fixed bottom-8 right-8 z-50">
          <Button
            variant="outline"
            size="sm"
            onClick={onSwitchToRedesign}
            className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-md shadow-xl border-2 hover:scale-105 transition-transform"
          >
            <Sparkles className="h-3.5 w-3.5 mr-2" />
            Switch to New Design
          </Button>
        </div>
      )}
      <div className="container mx-auto px-4 py-12 max-w-7xl">
        {/* Header */}
        <div className="relative text-center mb-8">
          {/* Theme Toggle - Positioned at top-right */}
          <div className="absolute right-0 top-0">
            <ThemeToggle />
          </div>
          
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-lg">
              <BarChart3 className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              docs-to-eval
            </h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Automated LLM Evaluation System with Math-Verify Integration
          </p>
        </div>

        {/* Main Content */}
        <Card className="shadow-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
          <Tabs defaultValue="evaluate" className="w-full">
            <div className="border-b border-gray-200 dark:border-gray-700 bg-white/50 dark:bg-slate-800/50 rounded-t-lg">
              <TabsList className="grid w-full grid-cols-4 bg-transparent h-auto p-4">
                <TabsTrigger 
                  value="evaluate" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white transition-colors"
                >
                  <FlaskConical className="h-4 w-4" />
                  Evaluate Corpus
                </TabsTrigger>
                <TabsTrigger 
                  value="classify" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white transition-colors"
                >
                  <Users className="h-4 w-4" />
                  Classify Corpus
                </TabsTrigger>
                <TabsTrigger 
                  value="runs" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white transition-colors"
                >
                  <BarChart3 className="h-4 w-4" />
                  Evaluation Runs
                </TabsTrigger>
                <TabsTrigger 
                  value="settings" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white transition-colors"
                >
                  <Settings className="h-4 w-4" />
                  Settings
                </TabsTrigger>
              </TabsList>
            </div>

            <div className="p-6">
              <TabsContent value="evaluate" className="mt-0">
                <EvaluationTab />
              </TabsContent>
              
              <TabsContent value="classify" className="mt-0">
                <ClassificationTab />
              </TabsContent>
              
              <TabsContent value="runs" className="mt-0">
                <RunsTab />
              </TabsContent>
              
              <TabsContent value="settings" className="mt-0">
                <SettingsTab />
              </TabsContent>
            </div>
          </Tabs>
        </Card>
      </div>
    </div>
  );
}
