// The panel as a webpack module-federation container, the shape Superset 7.0.0rc2 loads (`ExtensionsLoader.loadModule`
// in superset-frontend/src/extensions/ExtensionsLoader.ts): a remote entry that registers `agenticfabriq_mnemiq` on
// the window, exposing `./index`. Laid out as rc2's own extension template has it
// (superset-extensions-cli/src/superset_extensions_cli/templates/frontend/webpack.config.js.j2), with three changes:
// - `publicPath: "auto"`: chunks load from wherever the remote entry was served, so an application root in front of
//   Superset needs no rebuild;
// - React is required at ^18.3.0 (the host's), not the template's ^17.0.2;
// - no source map and no extracted licence file: the packer refuses a map, and every file must carry a hash.
const path = require("path");
// webpack's own minimizer (terser, by default), reached through webpack: pnpm keeps it a dependency of webpack's, not of
// this package's. webpack 5.112 configures it in `optimization.minimizer` as `minimizer-webpack-plugin`.
const MinimizerPlugin = require(
  require.resolve("minimizer-webpack-plugin", { paths: [path.dirname(require.resolve("webpack/package.json"))] }),
);
const { ModuleFederationPlugin } = require("webpack").container;
const packageConfig = require("./package.json");

// Shared with the host and never bundled: `import: false` leaves no fallback copy in the container.
// `requiredVersion: false` for `@apache-superset/core`, which rc2's template leaves unversioned: the loader injects the
// host's own under the host's version key, whatever this package's peer list says.
const host = (requiredVersion) => ({ singleton: true, import: false, requiredVersion });

module.exports = (env, argv) => {
  const isProd = argv.mode === "production";
  return {
    // In production the remote's only entry is the federation container, so no unhashed `main.js` is emitted.
    entry: isProd ? {} : "./src/index.tsx",
    mode: isProd ? "production" : "development",
    // ES2020 has `globalThis`, so webpack's runtime reads it as is, without its `new Function("return this")` fallback:
    // nothing in the bundle evaluates a string as code, whatever the host's content security policy allows.
    target: ["web", "es2020"],
    devtool: false,
    output: {
      clean: true,
      filename: isProd ? undefined : "[name].[contenthash].js",
      chunkFilename: "[name].[contenthash].js",
      path: path.resolve(__dirname, "dist"),
      publicPath: "auto",
    },
    resolve: { extensions: [".ts", ".tsx", ".js", ".jsx"] },
    module: {
      rules: [{ test: /\.tsx?$/, use: { loader: "ts-loader", options: { onlyCompileBundledFiles: true } } }],
    },
    optimization: { minimizer: [new MinimizerPlugin({ extractComments: false })] },
    plugins: [
      new ModuleFederationPlugin({
        name: "agenticfabriq_mnemiq",
        filename: "remoteEntry.[contenthash].js",
        exposes: { "./index": "./src/index.tsx" },
        shared: {
          react: host(packageConfig.peerDependencies.react),
          "react-dom": host(packageConfig.peerDependencies["react-dom"]),
          antd: host(packageConfig.peerDependencies.antd),
          "@apache-superset/core": host(false),
        },
      }),
    ],
  };
};
