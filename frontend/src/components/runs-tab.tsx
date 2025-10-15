"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { RefreshCw, Eye, Trash2, Settings, TrendingUp, Calendar, BarChart3, Zap } from "lucide-react";
import { toast } from "sonner";

interface EvaluationRun {
  run_id: string;
  run_name?: string;
  status: string;
  start_time: string;
  eval_type: string;
  num_questions: number;
  results?: {
    finetune_test_set?: {
      enabled: boolean;
      train_questions: number;
      test_questions: number;
    };
  };
  lora_finetuning?: {
    status: string;
    comparison_results?: {
      improvement: number;
      original_accuracy: number;
      finetuned_accuracy: number;
    };
  };
}

export function RunsTab() {
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const loadRuns = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/runs');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // Handle both formats: {runs: [...]} or [...]
      const runsList = Array.isArray(data) ? data : (data.runs || []);

      setRuns(runsList);
      toast.success(`Loaded ${runsList.length} evaluation runs`);

    } catch (error: any) {
      console.error('Failed to load runs:', error);
      toast.error(`Failed to load runs: ${error.message}`);
      setRuns([]);
    } finally {
      setIsLoading(false);
    }
  };

  const viewRun = async (runId: string) => {
    try {
      const response = await fetch(`/api/v1/evaluation/${runId}/results`);
      
      if (!response.ok) {
        throw new Error(`Failed to load run results: ${response.status}`);
      }

      const resultWrapper = await response.json();
      
      // For now, just show a success message and download results
      // In a full implementation, you'd navigate to a detailed results view
      toast.success('Run results loaded! Opening download...');
      window.open(`/api/v1/evaluation/${runId}/download`, '_blank');

    } catch (error: any) {
      console.error('Failed to load run:', error);
      toast.error('Failed to load run results');
    }
  };

  const deleteRun = async (runId: string) => {
    if (!confirm('Are you sure you want to delete this run? This action cannot be undone.')) {
      return;
    }

    try {
      const response = await fetch(`/api/v1/runs/${runId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        toast.success('Run deleted successfully');
        loadRuns(); // Refresh the list
      } else {
        toast.error('Failed to delete run');
      }

    } catch (error: any) {
      console.error('Failed to delete run:', error);
      toast.error('Failed to delete run: ' + error.message);
    }
  };

  const openFinetuningDashboard = (runId: string) => {
    const url = `/api/v1/evaluation/${runId}/lora-finetune/dashboard?run_id=${runId}`;
    window.open(url, '_blank');
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      completed: "default",
      running: "secondary",
      failed: "destructive",
      pending: "outline"
    };

    const colors: Record<string, string> = {
      completed: "bg-green-500",
      running: "bg-yellow-500",
      failed: "bg-red-500",
      pending: "bg-gray-500"
    };

    return (
      <Badge variant={variants[status] || "outline"} className="capitalize">
        <div className={`w-2 h-2 rounded-full mr-2 ${colors[status] || "bg-gray-500"}`}></div>
        {status}
      </Badge>
    );
  };

  const getFinetuneStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      completed: "default",
      running: "secondary",
      failed: "destructive",
      not_started: "outline"
    };

    return (
      <Badge variant={variants[status] || "outline"} className="text-xs">
        {status.replace('_', ' ')}
      </Badge>
    );
  };

  useEffect(() => {
    loadRuns();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Evaluation Runs</h2>
          <p className="text-muted-foreground">
            View and manage your evaluation runs and fine-tuning results
          </p>
        </div>
        <Button onClick={loadRuns} disabled={isLoading} className="flex items-center gap-2">
          <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh Runs
        </Button>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
            <span className="text-muted-foreground">Loading evaluation runs...</span>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && runs.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <BarChart3 className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No evaluation runs found</h3>
            <p className="text-muted-foreground text-center max-w-md">
              Start your first evaluation in the "Evaluate Corpus" tab to see results here.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Runs List */}
      {!isLoading && runs.length > 0 && (
        <div className="space-y-4">
          {runs.map((run) => {
            const finetuneStatus = run.lora_finetuning?.status || 'not_started';
            const hasFinetuneSet = run.results?.finetune_test_set?.enabled || false;

            return (
              <Card key={run.run_id} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    {/* Run Info */}
                    <div className="flex-1 space-y-3">
                      <div className="flex items-center gap-3">
                        <h3 className="text-lg font-semibold">
                          {run.run_name || run.run_id}
                        </h3>
                        {getStatusBadge(run.status)}
                      </div>

                      <div className="flex items-center gap-6 text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Calendar className="h-4 w-4" />
                          {new Date(run.start_time).toLocaleString()}
                        </div>
                        <div className="flex items-center gap-1">
                          <BarChart3 className="h-4 w-4" />
                          {run.eval_type.replace('_', ' ')}
                        </div>
                        <div className="flex items-center gap-1">
                          <TrendingUp className="h-4 w-4" />
                          {run.num_questions} questions
                        </div>
                      </div>

                      {hasFinetuneSet && (
                        <div className="text-sm text-muted-foreground">
                          <span className="font-medium">Fine-tune Split:</span> {run.results?.finetune_test_set?.train_questions} train + {run.results?.finetune_test_set?.test_questions} test
                        </div>
                      )}

                      {/* Fine-tuning Info */}
                      {hasFinetuneSet && run.status === 'completed' && (
                        <Card className="bg-muted/50 border-l-4 border-l-blue-500">
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <Zap className="h-4 w-4 text-blue-500" />
                                <span className="font-medium text-blue-700">LoRA Fine-tuning</span>
                              </div>
                              {getFinetuneStatusBadge(finetuneStatus)}
                            </div>
                            {finetuneStatus === 'completed' && run.lora_finetuning?.comparison_results && (
                              <div className="text-sm text-muted-foreground">
                                <span className="font-medium">Performance Improvement:</span>{' '}
                                <span className={`font-semibold ${run.lora_finetuning.comparison_results.improvement > 0 ? 'text-green-600' : 'text-red-600'}`}>
                                  {run.lora_finetuning.comparison_results.improvement > 0 ? '+' : ''}
                                  {run.lora_finetuning.comparison_results.improvement.toFixed(1)}%
                                </span>{' '}
                                ({(run.lora_finetuning.comparison_results.original_accuracy * 100).toFixed(1)}% → {(run.lora_finetuning.comparison_results.finetuned_accuracy * 100).toFixed(1)}%)
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex flex-col gap-2 min-w-[200px]">
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => viewRun(run.run_id)}
                          className="flex items-center gap-1 flex-1"
                        >
                          <Eye className="h-4 w-4" />
                          View Results
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteRun(run.run_id)}
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>

                      {hasFinetuneSet && run.status === 'completed' && (
                        <Button
                          onClick={() => openFinetuningDashboard(run.run_id)}
                          size="sm"
                          className="flex items-center gap-1 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
                        >
                          <Settings className="h-4 w-4" />
                          Fine-tune Model
                        </Button>
                      )}

                      {!hasFinetuneSet && run.status === 'completed' && (
                        <div className="text-xs text-muted-foreground text-center italic py-2">
                          Fine-tuning not available<br />(no test set created)
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Run ID */}
                  <div className="mt-4 pt-3 border-t">
                    <div className="text-xs text-muted-foreground">
                      Run ID: <code className="bg-muted px-1 py-0.5 rounded text-xs">{run.run_id}</code>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Statistics */}
      {!isLoading && runs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{runs.length}</div>
                <div className="text-sm text-muted-foreground">Total Runs</div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {runs.filter(r => r.status === 'completed').length}
                </div>
                <div className="text-sm text-muted-foreground">Completed</div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">
                  {runs.filter(r => r.status === 'running').length}
                </div>
                <div className="text-sm text-muted-foreground">Running</div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {runs.filter(r => r.results?.finetune_test_set?.enabled).length}
                </div>
                <div className="text-sm text-muted-foreground">With Fine-tuning</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}