/**
 * The ECharts instance, registered with only what we draw.
 *
 * The static named imports here are what make it tree-shake: a dynamic
 * `import("echarts/charts")` asks for the whole barrel and ships every chart type
 * in the library. This module is itself lazy-imported, so the cost lands in its own
 * chunk and only when a chart is first rendered.
 *
 * SVG rather than canvas: charts then render where there is no canvas -- jsdom under
 * test -- and stay crisp when scaled.
 */

import * as echarts from "echarts/core";
import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import { SVGRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  SVGRenderer,
]);

export default echarts;
