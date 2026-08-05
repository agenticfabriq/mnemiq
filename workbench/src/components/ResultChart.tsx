/**
 * The result, drawn.
 *
 * ECharts is lazy-loaded and registers only the three chart types we build, with
 * the SVG renderer -- so it stays out of the main bundle, and a chart still renders
 * where there is no canvas.
 *
 * Colours are read from the stylesheet rather than hardcoded, so the one place
 * light and dark are defined stays the one place they are defined.
 */

import { useEffect, useRef, useState } from "react";

import { buildOption, type Palette } from "../lib/chartoption";
import type { ChartSpec } from "../lib/chartspec";
import type { ResultPreview } from "../lib/types";

type ECharts = typeof import("../lib/echarts").default;
let loading: Promise<ECharts> | null = null;

const loadECharts = (): Promise<ECharts> =>
  (loading ??= import("../lib/echarts").then((m) => m.default));

function readPalette(el: HTMLElement): Palette {
  const style = getComputedStyle(el);
  const v = (name: string) => style.getPropertyValue(name).trim();
  return {
    series: Array.from({ length: 8 }, (_, i) => v(`--series-${i + 1}`)).filter(Boolean),
    ink: v("--ink"),
    graphite: v("--graphite"),
    rule: v("--rule"),
    surface: v("--surface"),
  };
}

export function ResultChart({
  spec,
  preview,
  measure,
}: {
  spec: ChartSpec;
  preview: ResultPreview;
  measure: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let chart: { setOption: (o: object) => void; resize: () => void; dispose: () => void } | null =
      null;
    let disposed = false;

    loadECharts()
      .then((echarts) => {
        if (disposed || !host.current) return;
        chart = echarts.init(host.current, undefined, { renderer: "svg" });
        chart.setOption(buildOption(spec, preview, readPalette(host.current), measure));
      })
      .catch(() => setFailed(true));

    // Re-read the palette when the theme flips; the option carries resolved colours.
    const observer = new MutationObserver(() => {
      if (chart && host.current) {
        chart.setOption(buildOption(spec, preview, readPalette(host.current), measure));
      }
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    const resize = new ResizeObserver(() => chart?.resize());
    resize.observe(el);

    return () => {
      disposed = true;
      observer.disconnect();
      resize.disconnect();
      chart?.dispose();
    };
  }, [spec, preview, measure]);

  if (failed) return null; // the table is always there; a missing chart is not an error

  return (
    <div
      ref={host}
      role="img"
      aria-label={`${spec.kind} chart of ${measure} by ${spec.labelColumn}`}
      className="h-64 w-full"
    />
  );
}
