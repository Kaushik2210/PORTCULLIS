import type { NextConfig } from "next";

// Static export for GitHub Pages (project site, served under /PORTCULLIS/).
// Only used for the public demo build - `npm run dev` and the CI job's
// `npm run build` are unaffected by basePath since GITHUB_PAGES is unset there.
const isGithubPages = process.env.GITHUB_PAGES === "true";

const nextConfig: NextConfig = {
  output: "export",
  basePath: isGithubPages ? "/PORTCULLIS" : "",
  images: { unoptimized: true },
};

export default nextConfig;
