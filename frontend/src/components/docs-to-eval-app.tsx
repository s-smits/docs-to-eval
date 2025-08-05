"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { EvaluationTab } from "./evaluation-tab";
import { ClassificationTab } from "./classification-tab";
import { RunsTab } from "./runs-tab";
import { SettingsTab } from "./settings-tab";
import { BarChart3, Users, Settings, FlaskConical } from "lucide-react";

export function DocsToEvalApp() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl">
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
        <Card className="shadow-2xl border-0 bg-white/70 backdrop-blur-sm">
          <Tabs defaultValue="evaluate" className="w-full">
            <div className="border-b bg-white/50 rounded-t-lg">
              <TabsList className="grid w-full grid-cols-4 bg-transparent h-auto p-4">
                <TabsTrigger 
                  value="evaluate" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
                >
                  <FlaskConical className="h-4 w-4" />
                  Evaluate Corpus
                </TabsTrigger>
                <TabsTrigger 
                  value="classify" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
                >
                  <Users className="h-4 w-4" />
                  Classify Corpus
                </TabsTrigger>
                <TabsTrigger 
                  value="runs" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
                >
                  <BarChart3 className="h-4 w-4" />
                  Evaluation Runs
                </TabsTrigger>
                <TabsTrigger 
                  value="settings" 
                  className="flex items-center gap-2 py-3 px-6 data-[state=active]:bg-blue-500 data-[state=active]:text-white"
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