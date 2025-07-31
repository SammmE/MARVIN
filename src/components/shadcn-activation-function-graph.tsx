import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from "recharts";

interface ActivationFunction {
    name: string;
    code: string;
}

interface ActivationFunctionGraphProps {
    functions: ActivationFunction[];
    currentFunction?: string;
    functionCode?: string; // For backward compatibility
    title?: string;
}

export function ShadcnActivationFunctionGraph({
    functions = [],
    functionCode,
    title = "Activation Function Visualization",
}: ActivationFunctionGraphProps) {
    const [data, setData] = useState<
        Array<{ x: number; y: number; function?: string }>
    >([]);
    const [error, setError] = useState<string | null>(null);
    const [domain, setDomain] = useState<[number, number]>([-2, 2]);

    // Extract function names or determine a name from the function code
    const extractFunctionName = (code: string): string => {
        const functionNameMatch = code.match(/function\s+([a-zA-Z0-9_]+)/);
        if (functionNameMatch && functionNameMatch[1]) {
            return functionNameMatch[1];
        }
        return "Custom Function";
    };

    useEffect(() => {
        try {
            // Create a safe function wrapper with limited context
            const createActivationFunction = new Function(
                "code",
                `
        try {
          // Extract just the function body if it's a full function declaration
          const funcMatch = code.match(/function\\s+\\w*\\s*\\(\\w+\\)\\s*{([\\s\\S]*)}/);
          const arrowFuncMatch = code.match(/\\(\\w+\\)\\s*=>\\s*{([\\s\\S]*)}/);
          const simpleTernaryMatch = code.match(/(\\w+)\\s*=>\\s*([^{][\\s\\S]*)/);
          
          let funcBody;
          let paramName = 'x';
          
          if (funcMatch) {
            funcBody = funcMatch[1];
            paramName = code.match(/function\\s+\\w*\\s*\\((\\w+)\\)/)[1] || 'x';
          } else if (arrowFuncMatch) {
            funcBody = arrowFuncMatch[1];
            paramName = code.match(/\\((\\w+)\\)\\s*=>/)[1] || 'x';
          } else if (simpleTernaryMatch) {
            funcBody = \`return \${simpleTernaryMatch[2]}\`;
            paramName = simpleTernaryMatch[1];
          } else {
            // Just try to use the code as is
            funcBody = code.includes('return') ? code : \`return \${code}\`;
          }
          
          // Create the function
          return new Function(paramName, funcBody);
        } catch (e) {
          throw new Error('Failed to parse function: ' + e.message);
        }
      `,
            );

            const allPoints: Array<{ x: number; y: number; function: string }> = [];
            const names: string[] = [];
            let minY = Infinity;
            let maxY = -Infinity;

            // Handle legacy single function code
            if (functionCode && (!functions || functions.length === 0)) {
                // Determine the function name
                const fnName = extractFunctionName(functionCode);
                names.push(fnName);

                // Generate points for this function
                const activationFunc = createActivationFunction(functionCode);
                for (let x = -5; x <= 5; x += 0.2) {
                    try {
                        const y = activationFunc(x);
                        if (typeof y === "number" && !isNaN(y) && isFinite(y)) {
                            allPoints.push({ x, y, function: fnName });
                            minY = Math.min(minY, y);
                            maxY = Math.max(maxY, y);
                        }
                    } catch (e) {
                        console.error(`Error calculating point for ${fnName} at x=${x}`, e);
                    }
                }
            }

            // Process multiple functions
            for (const func of functions) {
                try {
                    names.push(func.name);
                    const activationFunc = createActivationFunction(func.code);

                    for (let x = -5; x <= 5; x += 0.2) {
                        try {
                            const y = activationFunc(x);
                            if (typeof y === "number" && !isNaN(y) && isFinite(y)) {
                                allPoints.push({ x, y, function: func.name });
                                minY = Math.min(minY, y);
                                maxY = Math.max(maxY, y);
                            }
                        } catch (e) {
                            console.error(
                                `Error calculating point for ${func.name} at x=${x}`,
                                e,
                            );
                        }
                    }
                } catch (e) {
                    console.error(`Error processing function ${func.name}:`, e);
                }
            }

            // Set dynamic domain for Y axis
            const yPadding = Math.max(1, (maxY - minY) * 0.2);
            setDomain([Math.floor(minY - yPadding), Math.ceil(maxY + yPadding)]);

            setData(allPoints);
            setError(null);
        } catch (e) {
            console.error("Error rendering activation functions:", e);
            setError(
                e instanceof Error
                    ? e.message
                    : "Failed to parse or execute the functions",
            );
            setData([]);
        }
    }, [functions, functionCode]);

    if (error) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>{title}</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="text-red-500 p-4 bg-red-50 rounded-md">
                        Error: {error}
                    </div>
                </CardContent>
            </Card>
        );
    }

    // Group data by function name
    const groupedData = data.reduce<{
        [key: string]: Array<{ x: number; y: number }>;
    }>((acc, point) => {
        const funcName = point.function || "default";
        if (!acc[funcName]) {
            acc[funcName] = [];
        }
        acc[funcName].push({ x: point.x, y: point.y });
        return acc;
    }, {});

    // Get unique x values in ascending order
    const uniqueXValues = [...new Set(data.map((d) => d.x))].sort(
        (a, b) => a - b,
    );

    // Colors for different functions
    const colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#f97316",
        "#0891b2",
    ];

    return (
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                            data={uniqueXValues.map((x) => ({ x }))}
                            margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey="x"
                                domain={[-5, 5]}
                                tickCount={5}
                                type="number"
                                label={{
                                    value: "Input (x)",
                                    position: "insideBottom",
                                    offset: -5,
                                }}
                            />
                            <YAxis
                                domain={domain}
                                tickCount={5}
                                label={{
                                    value: "Output f(x)",
                                    angle: -90,
                                    position: "insideLeft",
                                    offset: 10,
                                }}
                            />
                            <Tooltip
                                formatter={(value: number, name: string) => [
                                    value.toFixed(4),
                                    name,
                                ]}
                                labelFormatter={(label: number) => `Input: ${label.toFixed(2)}`}
                                contentStyle={{
                                    backgroundColor: "var(--background)",
                                    borderColor: "var(--border)",
                                    borderRadius: "0.375rem",
                                }}
                            />
                            {Object.entries(groupedData).map(
                                ([funcName, funcData], index) => (
                                    <Line
                                        key={funcName}
                                        name={funcName}
                                        type="monotone"
                                        data={funcData}
                                        dataKey="y"
                                        stroke={colors[index % colors.length]}
                                        strokeWidth={2}
                                        dot={false}
                                        activeDot={{ r: 4 }}
                                        isAnimationActive={false}
                                    />
                                ),
                            )}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                {Object.keys(groupedData).length > 1 && (
                    <div className="flex flex-wrap gap-2 mt-2 items-center justify-center">
                        {Object.keys(groupedData).map((funcName, index) => (
                            <div key={funcName} className="flex items-center gap-1">
                                <div
                                    className="w-3 h-3 rounded-full"
                                    style={{ backgroundColor: colors[index % colors.length] }}
                                />
                                <span className="text-sm">{funcName}</span>
                            </div>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
