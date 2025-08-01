import { useState, useRef } from "react";
import { Button } from "./ui/button";
import { Upload } from "lucide-react";
import { useDataStore } from "@/lib/oscar-store";

export function FileUploader() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const { setDataset } = useDataStore();

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsLoading(true);
        setError(null);

        try {
            const fileExtension = file.name.split(".").pop()?.toLowerCase();
            if (fileExtension === "csv") {
                await handleCSVFile(file);
            } else if (fileExtension === "json") {
                await handleJSONFile(file);
            } else {
                throw new Error(
                    "Unsupported file format. Please upload a CSV or JSON file.",
                );
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Error loading file");
        } finally {
            setIsLoading(false);
            // Reset the file input value so the same file can be uploaded again if needed
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    const handleCSVFile = async (file: File) => {
        const text = await file.text();
        const lines = text.split(/\r?\n/).filter((line) => line.trim());

        if (lines.length === 0) {
            throw new Error("CSV file is empty");
        }

        // Parse header row
        const headers = lines[0].split(",").map((header) => header.trim());

        // Parse data rows
        const data = lines.slice(1).map((line) => {
            const values = line.split(",");
            const row: Record<string, unknown> = {};

            headers.forEach((header, index) => {
                const value = values[index]?.trim();
                if (value === undefined || value === "") {
                    row[header] = null;
                } else if (!isNaN(Number(value))) {
                    row[header] = Number(value);
                } else {
                    row[header] = value;
                }
            });

            return row;
        });

        setDataset({ data, columns: headers });
    };

    const handleJSONFile = async (file: File) => {
        const text = await file.text();
        const json = JSON.parse(text);

        if (!Array.isArray(json)) {
            throw new Error("JSON file must contain an array of objects");
        }

        if (json.length === 0) {
            throw new Error("JSON file contains empty array");
        }

        // Extract columns from the first object's keys
        const columns = Object.keys(json[0]);

        setDataset({ data: json, columns });
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();

        const file = e.dataTransfer.files[0];
        if (file) {
            const fileExtension = file.name.split(".").pop()?.toLowerCase();
            if (fileExtension === "csv" || fileExtension === "json") {
                const event = {
                    target: {
                        files: [file],
                    },
                } as unknown as React.ChangeEvent<HTMLInputElement>;
                handleFileChange(event);
            } else {
                setError("Unsupported file format. Please upload a CSV or JSON file.");
            }
        }
    };

    return (
        <div
            className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl p-6 text-center"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
        >
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-gray-100">
                Upload CSV or JSON
            </h3>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                Drag and drop a file or click to browse
            </p>
            <div className="mt-4">
                <Button onClick={handleUploadClick} disabled={isLoading}>
                    {isLoading ? "Loading..." : "Select File"}
                </Button>
                <input
                    type="file"
                    className="hidden"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept=".csv,.json"
                />
            </div>
            {error && <div className="mt-2 text-sm text-red-600">{error}</div>}
        </div>
    );
}
