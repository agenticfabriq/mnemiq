/**
 * Result rows -> an ECharts option.
 *
 * Built in code from a small spec, never authored by a model. A chart is a claim
 * about the data; a hand-built option is one we can test, a generated config is one
 * we would have to trust.
 *
 * **One measure per chart.** A SQL result routinely returns measures whose scales
 * are orders apart -- a claim total in cents beside a policy count -- and drawing
 * them on one axis buries the small one while a second axis is the single worst
 * thing a chart can do. So the reader picks the measure, and the chart has one
 * series, one scale, and one hue. That also means no legend: the caption names it.
 */

import { toNumber } from "./chartspec";
import type { ChartSpec } from "./chartspec";
import type { ResultPreview } from "./types";

export type Palette = {
  series: string[];
  ink: string;
  graphite: string;
  rule: string;
  surface: string;
};

type Option = Record<string, unknown>;

/**
 * Axis ticks for money in cents run to thirteen digits, which collide into an
 * unreadable column. Abbreviate the axis; the tooltip and the table still carry the
 * exact figure.
 */
export function compact(n: number): string {
  const abs = Math.abs(n);
  const step = (div: number, suffix: string) => {
    const v = n / div;
    return `${Number.isInteger(v) ? v : Number(v.toFixed(1))}${suffix}`;
  };
  if (abs >= 1e12) return step(1e12, "T");
  if (abs >= 1e9) return step(1e9, "B");
  if (abs >= 1e6) return step(1e6, "M");
  if (abs >= 1e4) return step(1e3, "k");
  return n.toLocaleString("en-US");
}

const exact = (n: number) => n.toLocaleString("en-US");

export function buildOption(
  spec: ChartSpec,
  preview: ResultPreview,
  palette: Palette,
  measure: string,
): Option {
  const index = (name: string) => preview.columns.indexOf(name);
  const labelAt = index(spec.labelColumn);
  const valueAt = index(measure);
  const color = palette.series[0] ?? "#2a78d6";
  const axisLabel = { color: palette.graphite, fontSize: 10, fontFamily: "inherit" };

  const labels = preview.rows.map((r) => String(r[labelAt] ?? ""));
  const values = preview.rows.map((r) => toNumber(r[valueAt] ?? null));

  const series =
    spec.kind === "line"
      ? {
          type: "line", data: values, name: measure,
          showSymbol: values.length <= 60, symbolSize: 8,
          lineStyle: { width: 2, color }, itemStyle: { color },
        }
      : spec.kind === "scatter"
        ? {
            type: "scatter", name: measure, symbolSize: 8, itemStyle: { color },
            data: preview.rows.map((r) => [
              toNumber(r[labelAt] ?? null),
              toNumber(r[valueAt] ?? null),
            ]),
          }
        : {
            type: "bar", data: values, name: measure,
            // Rounded data-end anchored to the baseline.
            itemStyle: { color, borderRadius: [4, 4, 0, 0] },
            barMaxWidth: 28,
          };

  const numericX = spec.kind === "scatter";
  return {
    animation: false,
    backgroundColor: "transparent",
    grid: { left: 8, right: 12, top: 12, bottom: numericX ? 30 : 8, containLabel: true },
    tooltip: {
      trigger: numericX ? "item" : "axis",
      axisPointer: { type: spec.kind === "line" ? "line" : "shadow" },
      backgroundColor: palette.surface,
      borderColor: palette.rule,
      textStyle: { color: palette.ink, fontSize: 11, fontFamily: "inherit" },
      valueFormatter: (v: unknown) => (typeof v === "number" ? exact(v) : String(v ?? "")),
    },
    xAxis: {
      type: numericX ? "value" : "category",
      ...(numericX ? {} : { data: labels }),
      ...(numericX
        ? {
            name: spec.labelColumn,
            nameLocation: "middle",
            nameGap: 24,
            nameTextStyle: { color: palette.graphite, fontSize: 10, fontFamily: "inherit" },
          }
        : {}),
      axisLine: { lineStyle: { color: palette.rule } },
      axisTick: { show: false },
      axisLabel: { ...axisLabel, hideOverlap: true },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { ...axisLabel, formatter: (v: number) => compact(v) },
      // Recessive grid: enough to read a value against, quiet enough to ignore.
      splitLine: { lineStyle: { color: palette.rule, opacity: 0.6 } },
    },
    series: [series],
  };
}
