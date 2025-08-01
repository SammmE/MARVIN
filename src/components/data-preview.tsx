import { useState } from "react";
import { useDataStore } from "@/lib/oscar-store";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { Button } from "./ui/button";

export function DataPreview() {
    const { dataset, stats } = useDataStore();
    const [showMore, setShowMore] = useState(false);

    if (!dataset || dataset.data.length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Data Preview</CardTitle>
                </CardHeader>
                <CardContent className="text-muted-foreground text-sm">
                    No data available
                </CardContent>
            </Card>
        );
    }

    const displayData = showMore ? dataset.data : dataset.data.slice(0, 10);

    return (
        <Card>
            <CardHeader>
                <CardTitle>Data Preview</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="overflow-auto max-h-[300px]">
                    <table className="w-full text-sm">
                        <thead className="bg-muted sticky top-0">
                            <tr>
                                {dataset.columns.map((column) => (
                                    <th
                                        key={column}
                                        className="px-3 py-2 text-left font-medium text-muted-foreground"
                                    >
                                        {column}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {displayData.map((row, rowIndex) => (
                                <tr
                                    key={rowIndex}
                                    className="border-b border-border/50 last:border-0"
                                >
                                    {dataset.columns.map((column) => (
                                        <td key={`${rowIndex}-${column}`} className="px-3 py-2">
                                            {row[column] !== null && row[column] !== undefined ? (
                                                String(row[column])
                                            ) : (
                                                <span className="text-muted-foreground">null</span>
                                            )}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {dataset.data.length > 10 && (
                    <div className="mt-4 flex justify-center">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setShowMore(!showMore)}
                        >
                            {showMore ? "Show Less" : "Show More"}
                        </Button>
                    </div>
                )}

                {stats && (
                    <div className="mt-6">
                        <h3 className="font-medium text-sm mb-2">Statistics</h3>
                        <div className="overflow-auto">
                            <table className="w-full text-xs">
                                <thead className="bg-muted">
                                    <tr>
                                        <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                                            Column
                                        </th>
                                        <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                                            Mean
                                        </th>
                                        <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                                            Std
                                        </th>
                                        <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                                            Min
                                        </th>
                                        <th className="px-3 py-2 text-left font-medium text-muted-foreground">
                                            Max
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {dataset.columns.map((column) => {
                                        // Skip if this column doesn't have numeric stats
                                        if (!stats.mean[column]) return null;

                                        return (
                                            <tr
                                                key={column}
                                                className="border-b border-border/50 last:border-0"
                                            >
                                                <td className="px-3 py-2 font-medium">{column}</td>
                                                <td className="px-3 py-2">
                                                    {stats.mean[column].toFixed(4)}
                                                </td>
                                                <td className="px-3 py-2">
                                                    {stats.std[column].toFixed(4)}
                                                </td>
                                                <td className="px-3 py-2">
                                                    {stats.min[column].toFixed(4)}
                                                </td>
                                                <td className="px-3 py-2">
                                                    {stats.max[column].toFixed(4)}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
